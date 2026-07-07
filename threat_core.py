"""
threat_core.py
══════════════════════════════════════════════════════════════════
Self-contained parallel threat-detection engine.

All three MediaPipe workers (Face, Pose, Hand) + WeaponDetector
+ ThreatFusion live in this single file.

Usage from Streamlit (or any caller):
    from threat_core import ThreatEngine

    engine = ThreatEngine(model_path="yolov8_updated.pt")
    engine.start()

    result: EngineResult = engine.process_frame(bgr_frame)
    engine.stop()
══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np

# ── optional YOLO import (graceful fallback for demo without model) ──
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════
# CONSTANTS / DEFAULTS
# ══════════════════════════════════════════════════════════════════

SAVE_DIR   = Path("logs/screenshots")
LOG_PATH   = Path("logs/events/events.jsonl")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

LOW_THRESH   = 0.30
MID_THRESH   = 0.55
HIGH_THRESH  = 0.75

W_WEAPON, W_FACE, W_BEHAVIOR, W_INTERACTION = 0.45, 0.20, 0.15, 0.20
MASKED_WEAPON_BONUS = 0.15
GRIP_BONUS          = 0.20
GRIP_IOU_THRESH     = 0.10
AUTO_SAVE_COOLDOWN  = 4

THREAT_WEIGHTS = {"gun": 1.0, "pistol": 1.0, "rifle": 1.0, "knife": 0.85}

# ══════════════════════════════════════════════════════════════════
# ENUMS / DATA CLASSES
# ══════════════════════════════════════════════════════════════════

class ThreatLevel(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"

LEVEL_COLOR = {
    ThreatLevel.LOW:    (60, 200, 50),
    ThreatLevel.MEDIUM: (0, 165, 255),
    ThreatLevel.HIGH:   (0, 0, 230),
}
LEVEL_HEX = {
    ThreatLevel.LOW:    "#22c55e",
    ThreatLevel.MEDIUM: "#f97316",
    ThreatLevel.HIGH:   "#ef4444",
}


@dataclass
class MPResult:
    face_score:        float = 0.0
    behavior_score:    float = 0.0
    interaction_score: float = 0.0
    masked_faces:      int   = 0
    active_grip:       bool  = False
    top_behavior:      str   = "normal"
    face_overlay:      Optional[np.ndarray] = field(default=None, repr=False)
    pose_overlay:      Optional[np.ndarray] = field(default=None, repr=False)
    hand_overlay:      Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class FusionResult:
    threat_level:      ThreatLevel
    threat_score:      float
    weapon_score:      float
    face_score:        float
    behavior_score:    float
    interaction_score: float
    weapons_found:     List[str] = field(default_factory=list)
    masked_faces:      int       = 0
    active_grip:       bool      = False
    top_behavior:      str       = "normal"

    def bgr(self) -> tuple:
        return LEVEL_COLOR[self.threat_level]

    def hex_color(self) -> str:
        return LEVEL_HEX[self.threat_level]


@dataclass
class EngineResult:
    """Everything the Streamlit UI needs from one processed frame."""
    fusion:         FusionResult
    annotated_frame: np.ndarray          # BGR, ready for st.image
    fps:            float
    timestamp:      str
    log_entries:    List[dict] = field(default_factory=list)

# ══════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════

def _iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)

# ══════════════════════════════════════════════════════════════════
# WEAPON DETECTOR
# ══════════════════════════════════════════════════════════════════

class WeaponDetector:
    def __init__(self, model_path: str, conf: float = 0.35, device: str = "cpu"):
        self.conf   = conf
        self.device = device
        self._model = None
        self._names: dict = {}

        if not _YOLO_AVAILABLE:
            print("[WeaponDetector] ultralytics not installed — running in DEMO mode.")
            return

        model_file = Path(model_path)
        if not model_file.exists():
            print(f"[WeaponDetector] Model file not found: {model_path}. DEMO mode.")
            return

        print(f"[WeaponDetector] Loading {model_path} …")
        self._model = YOLO(model_path)
        self._names = self._model.model.names
        print(f"[WeaponDetector] Classes: {list(self._names.values())}")

    # ── public API ──────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[dict]:
        if self._model is None:
            return []
        results = self._model.predict(
            frame, conf=self.conf, device=self.device,
            verbose=False, iou=0.45,
        )
        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                cls   = int(box.cls[0])
                label = self._names.get(cls, "unknown").lower()
                conf  = float(box.conf[0])
                bbox  = box.xyxy[0].cpu().numpy().astype(int).tolist()
                tw    = THREAT_WEIGHTS.get(label, 0.85)
                out.append({"label": label, "conf": conf, "bbox": bbox, "tw": tw})
        return out

    def weapon_score(self, detections: List[dict]) -> float:
        if not detections:
            return 0.0
        top = max(detections, key=lambda d: d["conf"] * d["tw"])
        return min(1.0, top["conf"] * top["tw"] + 0.05 * (len(detections) - 1))

    def draw(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            color = (0, 0, 230) if d["tw"] >= 0.90 else (0, 140, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{d['label']}  {d['conf']:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)),
                          (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, text, (x1 + 4, max(th, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return frame

# ══════════════════════════════════════════════════════════════════
# MEDIAPIPE THREAD BASE
# ══════════════════════════════════════════════════════════════════

class _MPThread(threading.Thread):
    def __init__(self, name: str):
        super().__init__(name=name, daemon=True)
        self.in_q:  queue.Queue = queue.Queue(maxsize=2)
        self.out_q: queue.Queue = queue.Queue(maxsize=2)
        self._stop_evt = threading.Event()
        self._paused   = threading.Event()
        self._paused.set()

    def stop(self):   self._stop_evt.set()
    def pause(self):  self._paused.clear()
    def resume(self): self._paused.set()

    def submit(self, frame: np.ndarray, aux=None):
        try:
            self.in_q.put_nowait((frame, aux))
        except queue.Full:
            pass

    def latest(self) -> Optional[dict]:
        result = None
        while True:
            try:
                result = self.out_q.get_nowait()
            except queue.Empty:
                break
        return result

    def _process(self, frame: np.ndarray, aux) -> dict:
        raise NotImplementedError

    def run(self):
        while not self._stop_evt.is_set():
            self._paused.wait()
            try:
                frame, aux = self.in_q.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                result = self._process(frame, aux)
                try:
                    self.out_q.put_nowait(result)
                except queue.Full:
                    try: self.out_q.get_nowait()
                    except queue.Empty: pass
                    self.out_q.put_nowait(result)
            except Exception as e:
                print(f"[{self.name}] Error: {e}")

# ══════════════════════════════════════════════════════════════════
# T1 — FACE ANALYSER
# ══════════════════════════════════════════════════════════════════

class FaceAnalyserThread(_MPThread):
    _LOWER_FACE_IDX = list(range(11, 18)) + list(range(57, 68))

    def __init__(self):
        super().__init__("FaceAnalyser")
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=8,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._draw_util = mp.solutions.drawing_utils
        self._draw_spec = mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 220, 255), thickness=1, circle_radius=1)

    def _process(self, frame: np.ndarray, aux) -> dict:
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = self._face_mesh.process(rgb)
        overlay = frame.copy()

        masked_count = 0
        total_faces  = 0
        face_score   = 0.0

        if res.multi_face_landmarks:
            total_faces = len(res.multi_face_landmarks)
            for fl in res.multi_face_landmarks:
                self._draw_util.draw_landmarks(
                    overlay, fl,
                    mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    self._draw_spec, self._draw_spec)

                lms   = fl.landmark
                lower = [lms[i] for i in self._LOWER_FACE_IDX if i < len(lms)]
                if lower:
                    vis_scores = []
                    for lm in lower:
                        px, py = int(lm.x * w), int(lm.y * h)
                        if 0 <= py < h and 0 <= px < w:
                            roi = frame[max(0, py - 4):py + 4, max(0, px - 4):px + 4]
                            vis_scores.append(float(np.std(roi)) if roi.size else 0)
                    avg_vis = np.mean(vis_scores) if vis_scores else 0
                    if avg_vis < 18:
                        masked_count += 1
                        x_coords = [int(lm.x * w) for lm in lms]
                        y_coords = [int(lm.y * h) for lm in lms]
                        if x_coords and y_coords:
                            fx1 = max(0, min(x_coords) - 10)
                            fy1 = max(0, min(y_coords) - 10)
                            fx2 = min(w, max(x_coords) + 10)
                            fy2 = min(h, max(y_coords) + 10)
                            cv2.rectangle(overlay, (fx1, fy1), (fx2, fy2), (0, 0, 200), 2)
                            cv2.putText(overlay, "MASKED", (fx1, fy1 - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)

        if total_faces > 0:
            face_score = min(1.0, masked_count / total_faces
                             + 0.1 * min(masked_count, 3))

        return {
            "face_score":   face_score,
            "masked_faces": masked_count,
            "overlay":      overlay,
        }

# ══════════════════════════════════════════════════════════════════
# T2 — POSE ANALYSER
# ══════════════════════════════════════════════════════════════════

class PoseAnalyserThread(_MPThread):
    _LM = mp.solutions.pose.PoseLandmark

    def __init__(self):
        super().__init__("PoseAnalyser")
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._draw_util  = mp.solutions.drawing_utils
        self._prev_lms   = None
        self._motion_buf = deque(maxlen=8)

    def _process(self, frame: np.ndarray, aux) -> dict:
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = self._pose.process(rgb)
        overlay = frame.copy()

        behavior_score = 0.0
        top_behavior   = "normal"

        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            self._draw_util.draw_landmarks(
                overlay, res.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                self._draw_util.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
                self._draw_util.DrawingSpec(color=(0, 200, 100), thickness=2))

            def y(idx): return lms[idx.value].y if idx.value < len(lms) else 0.5
            def v(idx): return lms[idx.value].visibility if idx.value < len(lms) else 0.0

            L = self._LM
            arms_raised = 0
            if v(L.LEFT_WRIST)  > 0.5 and y(L.LEFT_WRIST)  < y(L.LEFT_SHOULDER):
                arms_raised += 1
            if v(L.RIGHT_WRIST) > 0.5 and y(L.RIGHT_WRIST) < y(L.RIGHT_SHOULDER):
                arms_raised += 1
            arm_score = arms_raised * 0.35

            hip_y  = (y(L.LEFT_HIP)  + y(L.RIGHT_HIP))  / 2
            knee_y = (y(L.LEFT_KNEE) + y(L.RIGHT_KNEE)) / 2
            crouch_score = max(0.0, 0.5 - abs(hip_y - knee_y)) * 0.5

            cur_arr = np.array([[lm.x, lm.y] for lm in lms])
            motion_score = 0.0
            if self._prev_lms is not None and self._prev_lms.shape == cur_arr.shape:
                delta = np.linalg.norm(cur_arr - self._prev_lms, axis=1).mean()
                self._motion_buf.append(delta)
                motion_score = min(1.0, np.mean(self._motion_buf) * 8)
            self._prev_lms = cur_arr

            behavior_score = min(1.0, arm_score + crouch_score + motion_score * 0.3)

            if arm_score > 0.5:
                top_behavior = "arms_raised"
            elif crouch_score > 0.15:
                top_behavior = "crouching"
            elif motion_score > 0.4:
                top_behavior = "rapid_movement"
            else:
                top_behavior = "normal"

            cv2.putText(overlay, f"POSE: {top_behavior.upper()} ({behavior_score:.0%})",
                        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)

        return {
            "behavior_score": behavior_score,
            "top_behavior":   top_behavior,
            "overlay":        overlay,
        }

# ══════════════════════════════════════════════════════════════════
# T3 — HAND / WEAPON INTERACTION ANALYSER
# ══════════════════════════════════════════════════════════════════

class HandAnalyserThread(_MPThread):
    def __init__(self):
        super().__init__("HandAnalyser")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._draw_util = mp.solutions.drawing_utils

    def _hand_bbox(self, hand_lms, h: int, w: int) -> List[int]:
        xs  = [int(lm.x * w) for lm in hand_lms.landmark]
        ys  = [int(lm.y * h) for lm in hand_lms.landmark]
        pad = 15
        return [max(0, min(xs) - pad), max(0, min(ys) - pad),
                min(w, max(xs) + pad), min(h, max(ys) + pad)]

    def _process(self, frame: np.ndarray, aux) -> dict:
        weapon_bboxes: List[List[int]] = aux or []
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = self._hands.process(rgb)
        overlay = frame.copy()

        active_grip       = False
        interaction_score = 0.0

        if res.multi_hand_landmarks:
            for hand_lms in res.multi_hand_landmarks:
                self._draw_util.draw_landmarks(
                    overlay, hand_lms, mp.solutions.hands.HAND_CONNECTIONS,
                    self._draw_util.DrawingSpec(color=(255, 180, 0), thickness=2, circle_radius=3),
                    self._draw_util.DrawingSpec(color=(200, 140, 0), thickness=2))

                h_bbox   = self._hand_bbox(hand_lms, h, w)
                best_iou = 0.0
                for wb in weapon_bboxes:
                    best_iou = max(best_iou, _iou(h_bbox, wb))

                if best_iou > GRIP_IOU_THRESH:
                    active_grip       = True
                    interaction_score = min(1.0, best_iou * 5)
                    cv2.rectangle(overlay,
                                  (h_bbox[0], h_bbox[1]), (h_bbox[2], h_bbox[3]),
                                  (0, 0, 255), 3)
                    cv2.putText(overlay, f"GRIP! iou={best_iou:.2f}",
                                (h_bbox[0], h_bbox[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                else:
                    cv2.rectangle(overlay,
                                  (h_bbox[0], h_bbox[1]), (h_bbox[2], h_bbox[3]),
                                  (255, 180, 0), 1)

        return {
            "interaction_score": interaction_score,
            "active_grip":       active_grip,
            "overlay":           overlay,
        }

# ══════════════════════════════════════════════════════════════════
# THREAT FUSION
# ══════════════════════════════════════════════════════════════════

def _fuse(ws: float, mp_res: MPResult, detections: List[dict]) -> FusionResult:
    raw = (ws                        * W_WEAPON
         + mp_res.face_score         * W_FACE
         + mp_res.behavior_score     * W_BEHAVIOR
         + mp_res.interaction_score  * W_INTERACTION)

    if ws > 0.30 and mp_res.masked_faces > 0:
        raw += MASKED_WEAPON_BONUS
    if mp_res.active_grip:
        raw += GRIP_BONUS

    score = float(min(raw, 1.0))

    if score >= HIGH_THRESH:
        level = ThreatLevel.HIGH
    elif score >= MID_THRESH:
        level = ThreatLevel.MEDIUM
    elif score >= LOW_THRESH:
        level = ThreatLevel.MEDIUM
    else:
        level = ThreatLevel.LOW

    return FusionResult(
        threat_level      = level,
        threat_score      = score,
        weapon_score      = ws,
        face_score        = mp_res.face_score,
        behavior_score    = mp_res.behavior_score,
        interaction_score = mp_res.interaction_score,
        weapons_found     = [d["label"] for d in detections],
        masked_faces      = mp_res.masked_faces,
        active_grip       = mp_res.active_grip,
        top_behavior      = mp_res.top_behavior,
    )

# ══════════════════════════════════════════════════════════════════
# HUD
# ══════════════════════════════════════════════════════════════════

def _draw_hud(frame: np.ndarray, fusion: FusionResult,
              fps: float, conf: float, paused: bool) -> np.ndarray:
    h, w  = frame.shape[:2]
    color = fusion.bgr()
    level = fusion.threat_level.value

    cv2.rectangle(frame, (0, 0), (w, 56), (12, 14, 20), -1)
    cv2.putText(frame, f"THREAT: {level}",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    info = f"FPS:{fps:.1f}  CONF:{conf:.2f}"
    if paused:
        info = "[PAUSED]  " + info
    cv2.putText(frame, info, (w - 240, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    cv2.rectangle(frame, (0, 56), (int(w * fusion.threat_score), 64), color, -1)
    cv2.rectangle(frame, (0, 56), (w, 64), (40, 40, 40), 1)

    sub_lines = [
        f"W:{fusion.weapon_score:.0%}",
        f"F:{fusion.face_score:.0%}",
        f"P:{fusion.behavior_score:.0%}",
        f"I:{fusion.interaction_score:.0%}",
    ]
    for i, txt in enumerate(sub_lines):
        cv2.putText(frame, txt, (w - 110, 18 + i * 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

    if fusion.weapons_found:
        seen  = list(dict.fromkeys(fusion.weapons_found))
        wtext = "Detected: " + ", ".join(seen)
        cv2.putText(frame, wtext, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)

    if fusion.active_grip:
        cv2.rectangle(frame, (0, 90 if not fusion.weapons_found else 100),
                      (w, 120), (0, 0, 180), -1)
        cv2.putText(frame,
                    "!!! WEAPON GRIP CONFIRMED — IMMEDIATE THREAT !!!",
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    if level == "HIGH" and int(time.time() * 3) % 2:
        cv2.rectangle(frame, (3, 67), (w - 3, h - 32), (0, 0, 230), 4)
        cv2.putText(frame, "!! WEAPON DETECTED !!",
                    (w // 2 - 175, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 230), 3)

    if fusion.top_behavior != "normal":
        cv2.putText(frame, f"BEHAVIOUR: {fusion.top_behavior.upper()}",
                    (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 180), 2)

    cv2.rectangle(frame, (0, h - 28), (w, h), (12, 14, 20), -1)
    cv2.putText(frame, "Parallel Threat Detection | Real-time",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (110, 110, 110), 1)
    return frame


def _blend_overlays(base: np.ndarray,
                    overlays: List[Optional[np.ndarray]],
                    alpha: float = 0.55) -> np.ndarray:
    result = base.copy()
    for ov in overlays:
        if ov is not None and ov.shape == base.shape:
            diff_mask = np.any(ov != base, axis=2)
            result[diff_mask] = cv2.addWeighted(
                base, 1 - alpha, ov, alpha, 0)[diff_mask]
    return result

# ══════════════════════════════════════════════════════════════════
# ALERT / LOGGING
# ══════════════════════════════════════════════════════════════════

class AlertManager:
    def __init__(self):
        self._cooldowns: Dict[str, float] = {}

    def process(self, fusion: FusionResult, frame: np.ndarray,
                cam_id: str = "cam_0") -> Optional[dict]:
        if fusion.threat_level == ThreatLevel.LOW:
            return None
        now  = time.time()
        last = self._cooldowns.get(cam_id, 0.0)
        if now - last < AUTO_SAVE_COOLDOWN:
            return None
        self._cooldowns[cam_id] = now
        shot = self._save_shot(frame, fusion, cam_id)
        ev   = self._log(fusion, cam_id, shot)
        return ev

    def _save_shot(self, frame: np.ndarray, f: FusionResult, cam_id: str) -> str:
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{cam_id}_{f.threat_level.value}_{ts}.jpg"
        path = SAVE_DIR / name
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return str(path)

    def _log(self, f: FusionResult, cam_id: str, shot: str) -> dict:
        ev = {
            "camera_id":        cam_id,
            "threat_level":     f.threat_level.value,
            "threat_score":     round(f.threat_score, 3),
            "weapon_score":     round(f.weapon_score, 3),
            "face_score":       round(f.face_score, 3),
            "behavior_score":   round(f.behavior_score, 3),
            "interaction_score":round(f.interaction_score, 3),
            "weapons_found":    f.weapons_found,
            "masked_faces":     f.masked_faces,
            "active_grip":      f.active_grip,
            "top_behavior":     f.top_behavior,
            "screenshot":       shot,
            "timestamp":        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        try:
            with open(LOG_PATH, "a") as fp:
                fp.write(json.dumps(ev) + "\n")
        except Exception as e:
            print(f"[LOG] Write failed: {e}")
        return ev

# ══════════════════════════════════════════════════════════════════
# PUBLIC ENGINE  ← the only thing callers need
# ══════════════════════════════════════════════════════════════════

class ThreatEngine:
    """
    One object that owns the detector + all three MP threads.

    Usage:
        engine = ThreatEngine(model_path="yolov8_updated.pt")
        engine.start()
        result = engine.process_frame(bgr_frame)
        engine.stop()
    """

    def __init__(self,
                 model_path: str   = "yolov8_updated.pt",
                 conf: float       = 0.35,
                 device: str       = "cpu"):
        self.conf      = conf
        self._detector = WeaponDetector(model_path, conf, device)
        self._alert    = AlertManager()
        self._t_face   = FaceAnalyserThread()
        self._t_pose   = PoseAnalyserThread()
        self._t_hand   = HandAnalyserThread()
        self._mp_cache = MPResult()
        self._fps      = 0.0
        self._prev_t   = time.time()
        self._paused   = False
        self._log_buf: List[dict] = []

    # ── lifecycle ───────────────────────────────────────────────

    def start(self):
        for t in (self._t_face, self._t_pose, self._t_hand):
            t.start()

    def stop(self):
        for t in (self._t_face, self._t_pose, self._t_hand):
            t.stop()

    def pause_mp(self):
        self._paused = True
        for t in (self._t_face, self._t_pose, self._t_hand):
            t.pause()

    def resume_mp(self):
        self._paused = False
        for t in (self._t_face, self._t_pose, self._t_hand):
            t.resume()

    def set_conf(self, conf: float):
        self.conf = conf
        self._detector.conf = conf

    # ── main API ────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> EngineResult:
        """Process one BGR frame; returns EngineResult."""
        frame_copy = frame.copy()

        # 1. Weapon detection (blocking, main thread)
        detections    = self._detector.detect(frame)
        weapon_bboxes = [d["bbox"] for d in detections]
        ws            = self._detector.weapon_score(detections)

        # 2. Submit to MP threads (non-blocking)
        self._t_face.submit(frame_copy)
        self._t_pose.submit(frame_copy)
        self._t_hand.submit(frame_copy, weapon_bboxes)

        # 3. Collect latest MP results
        face_res = self._t_face.latest()
        pose_res = self._t_pose.latest()
        hand_res = self._t_hand.latest()

        if face_res:
            self._mp_cache.face_score   = face_res["face_score"]
            self._mp_cache.masked_faces = face_res["masked_faces"]
            self._mp_cache.face_overlay = face_res["overlay"]
        if pose_res:
            self._mp_cache.behavior_score = pose_res["behavior_score"]
            self._mp_cache.top_behavior   = pose_res["top_behavior"]
            self._mp_cache.pose_overlay   = pose_res["overlay"]
        if hand_res:
            self._mp_cache.interaction_score = hand_res["interaction_score"]
            self._mp_cache.active_grip       = hand_res["active_grip"]
            self._mp_cache.hand_overlay      = hand_res["overlay"]

        # 4. Fusion
        fusion = _fuse(ws, self._mp_cache, detections)

        # 5. Draw weapon boxes
        annotated = self._detector.draw(frame.copy(), detections)

        # 6. Blend MP overlays
        annotated = _blend_overlays(
            annotated,
            [self._mp_cache.face_overlay,
             self._mp_cache.pose_overlay,
             self._mp_cache.hand_overlay],
            alpha=0.50,
        )

        # 7. FPS
        now        = time.time()
        self._fps  = 0.9 * self._fps + 0.1 / max(now - self._prev_t, 1e-6)
        self._prev_t = now

        # 8. HUD
        annotated = _draw_hud(annotated, fusion, self._fps, self.conf, self._paused)

        # 9. Alert
        ev = self._alert.process(fusion, annotated)
        logs: List[dict] = []
        if ev:
            self._log_buf.append(ev)
            logs = [ev]

        return EngineResult(
            fusion          = fusion,
            annotated_frame = annotated,
            fps             = self._fps,
            timestamp       = datetime.now().strftime("%H:%M:%S"),
            log_entries     = logs,
        )

    @property
    def log_history(self) -> List[dict]:
        return list(self._log_buf)