"""
app.py  —  Streamlit Demo for Parallel Threat Detection
══════════════════════════════════════════════════════════════════
Run:
    streamlit run app.py

Features
────────
  • Webcam live-feed (via st.camera_input) + optional video-file upload
  • Real-time threat scores, level badge, and score bars
  • Live annotated frame rendered frame-by-frame
  • Alert log table (auto-updated)
  • Sidebar controls: model path, confidence, pause MediaPipe, reset log
══════════════════════════════════════════════════════════════════
"""

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from threat_core import ThreatEngine, ThreatLevel, LEVEL_HEX

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Threat Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* Dark theme base */
body, .stApp { background-color: #0d0f14; color: #e2e8f0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #13161f;
    border-right: 1px solid #1e2436;
}

/* Threat badge */
.threat-badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 6px;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    font-family: 'Courier New', monospace;
}
.badge-LOW    { background:#14532d; color:#22c55e; border:1px solid #22c55e; }
.badge-MEDIUM { background:#431407; color:#f97316; border:1px solid #f97316; }
.badge-HIGH   { background:#450a0a; color:#ef4444; border:1px solid #ef4444;
                animation: pulse 0.6s infinite alternate; }
@keyframes pulse { from { opacity:1; } to { opacity:0.55; } }

/* Score bar track */
.score-track {
    background: #1e2436;
    border-radius: 4px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    margin: 2px 0 6px 0;
}
.score-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.2s ease;
}

/* Metric tile */
.metric-tile {
    background: #13161f;
    border: 1px solid #1e2436;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
}
.metric-value { font-size: 1.6rem; font-weight: 700; font-family: monospace; }
.metric-label { font-size: 0.72rem; color: #94a3b8; margin-top: 2px; }

/* Alert table */
.alert-row-HIGH   { color: #ef4444; }
.alert-row-MEDIUM { color: #f97316; }
.alert-row-LOW    { color: #22c55e; }

/* Section headers */
.section-head {
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 6px;
    border-bottom: 1px solid #1e2436;
    padding-bottom: 4px;
}

/* Override Streamlit top padding */
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "engine":          None,
        "engine_conf":     0.35,
        "engine_model":    "yolov8_updated.pt",
        "mp_paused":       False,
        "alert_log":       [],
        "last_fusion":     None,
        "frame_count":     0,
        "use_webcam":      True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ══════════════════════════════════════════════════════════════════
# ENGINE HELPERS
# ══════════════════════════════════════════════════════════════════

def get_engine() -> ThreatEngine:
    """Return the singleton engine, (re-)creating if needed."""
    s = st.session_state
    needs_new = (
        s["engine"] is None
        or s["engine"].conf != s["engine_conf"]
    )
    if needs_new:
        if s["engine"] is not None:
            s["engine"].stop()
        eng = ThreatEngine(
            model_path=s["engine_model"],
            conf=s["engine_conf"],
            device="cpu",
        )
        eng.start()
        s["engine"] = eng
    return s["engine"]


def decode_frame(img_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes → BGR numpy array."""
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    # Model path
    model_path = st.text_input(
        "YOLOv8 model path",
        value=st.session_state["engine_model"],
        help="Absolute or relative path to your .pt file. "
             "If the file doesn't exist the system runs in demo mode (no weapon detection).",
    )
    if model_path != st.session_state["engine_model"]:
        st.session_state["engine_model"] = model_path
        st.session_state["engine"] = None   # force reinit

    # Confidence
    conf = st.slider(
        "Detection confidence",
        min_value=0.10, max_value=0.95, step=0.05,
        value=st.session_state["engine_conf"],
    )
    if conf != st.session_state["engine_conf"]:
        st.session_state["engine_conf"] = conf
        if st.session_state["engine"]:
            st.session_state["engine"].set_conf(conf)

    st.divider()
    st.markdown("## 🎥 Input Source")
    use_webcam = st.toggle("Use live webcam", value=st.session_state["use_webcam"])
    st.session_state["use_webcam"] = use_webcam

    uploaded_video = None
    if not use_webcam:
        uploaded_video = st.file_uploader(
            "Upload video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a video to run threat detection on.",
        )

    st.divider()
    st.markdown("## 🧠 MediaPipe Controls")
    mp_pause_btn = st.button(
        "⏸  Pause MediaPipe" if not st.session_state["mp_paused"]
        else "▶  Resume MediaPipe",
        use_container_width=True,
    )
    if mp_pause_btn:
        eng = st.session_state.get("engine")
        if eng:
            if st.session_state["mp_paused"]:
                eng.resume_mp()
                st.session_state["mp_paused"] = False
            else:
                eng.pause_mp()
                st.session_state["mp_paused"] = True

    st.divider()
    st.markdown("## 📋 Alert Log")
    if st.button("🗑  Clear log", use_container_width=True):
        st.session_state["alert_log"] = []
        if st.session_state["engine"]:
            st.session_state["engine"]._log_buf.clear()

    st.markdown("## ℹ️ Fusion weights")
    st.markdown("""
| Module | Weight |
|--------|--------|
| Weapon | 0.45 |
| Face / mask | 0.20 |
| Pose / behaviour | 0.15 |
| Hand–weapon grip | 0.20 |

Bonuses: +0.15 masked+weapon, +0.20 active grip
""")

# ══════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════

st.markdown("# 🔍 Parallel Threat Detection System")
st.markdown(
    '<p style="color:#64748b;font-size:0.85rem;">Real-time weapon + behaviour analysis · '
    'YOLOv8 · MediaPipe Face / Pose / Hands · Parallel threads</p>',
    unsafe_allow_html=True,
)

col_feed, col_dash = st.columns([3, 2], gap="large")

# ── Left: video feed + camera input ────────────────────────────
with col_feed:
    st.markdown('<div class="section-head">Live Feed</div>', unsafe_allow_html=True)

    if use_webcam:
        # st.camera_input gives one snapshot per click/auto-refresh
        cam_image = st.camera_input(
            label="Webcam",
            label_visibility="collapsed",
        )
        frame_source = cam_image
    else:
        frame_source = None

    annotated_placeholder = st.empty()
    fps_placeholder       = st.empty()

# ── Right: dashboard ────────────────────────────────────────────
with col_dash:
    st.markdown('<div class="section-head">Threat Dashboard</div>', unsafe_allow_html=True)

    badge_ph      = st.empty()
    score_ph      = st.empty()
    breakdown_ph  = st.empty()
    grip_ph       = st.empty()
    behaviour_ph  = st.empty()

    st.divider()
    st.markdown('<div class="section-head">Alert Log (this session)</div>',
                unsafe_allow_html=True)
    log_ph = st.empty()

# ══════════════════════════════════════════════════════════════════
# HELPERS FOR RENDERING DASHBOARD
# ══════════════════════════════════════════════════════════════════

def _score_bar(label: str, value: float, color: str) -> str:
    pct = int(value * 100)
    return f"""
    <div style="margin-bottom:8px;">
      <div style="display:flex;justify-content:space-between;
                  font-size:0.78rem;margin-bottom:2px;">
        <span>{label}</span>
        <span style="color:{color};font-weight:600;">{pct}%</span>
      </div>
      <div class="score-track">
        <div class="score-fill"
             style="width:{pct}%;background:{color};"></div>
      </div>
    </div>"""


def render_dashboard(fusion, fps: float):
    level   = fusion.threat_level.value
    hx      = fusion.hex_color()
    cls_map = {"LOW": "badge-LOW", "MEDIUM": "badge-MEDIUM", "HIGH": "badge-HIGH"}

    # Badge
    badge_ph.markdown(
        f'<span class="threat-badge {cls_map[level]}">{level}</span>'
        f'<span style="font-size:0.82rem;color:#64748b;margin-left:12px;">'
        f'score {fusion.threat_score:.0%}</span>',
        unsafe_allow_html=True,
    )

    # Overall score bar
    score_ph.markdown(
        _score_bar("Overall threat score", fusion.threat_score, hx),
        unsafe_allow_html=True,
    )

    # Sub-score breakdown
    breakdown_html = (
        _score_bar("🔫 Weapon",       fusion.weapon_score,      "#ef4444")
        + _score_bar("😷 Face / mask", fusion.face_score,        "#f97316")
        + _score_bar("🏃 Behaviour",   fusion.behavior_score,    "#eab308")
        + _score_bar("✋ Grip / interact", fusion.interaction_score, "#a855f7")
    )
    breakdown_ph.markdown(breakdown_html, unsafe_allow_html=True)

    # Grip alert
    if fusion.active_grip:
        grip_ph.error("✋ WEAPON GRIP CONFIRMED — IMMEDIATE THREAT")
    else:
        grip_ph.empty()

    # Behaviour + weapons
    lines = []
    if fusion.weapons_found:
        seen = list(dict.fromkeys(fusion.weapons_found))
        lines.append(f"**Weapons detected:** {', '.join(seen)}")
    if fusion.masked_faces:
        lines.append(f"**Masked faces:** {fusion.masked_faces}")
    if fusion.top_behavior != "normal":
        lines.append(f"**Behaviour:** {fusion.top_behavior.replace('_',' ').title()}")
    if lines:
        behaviour_ph.warning("  \n".join(lines))
    else:
        behaviour_ph.empty()


def render_log(log: list):
    if not log:
        log_ph.caption("No alerts yet.")
        return
    rows = []
    for ev in reversed(log[-20:]):    # newest first, last 20
        lvl   = ev.get("threat_level", "")
        color = LEVEL_HEX.get(ThreatLevel(lvl) if lvl in ThreatLevel._value2member_map_ else ThreatLevel.LOW, "#94a3b8")
        rows.append(
            f'<tr style="color:{color};">'
            f'<td>{ev.get("timestamp","")}</td>'
            f'<td><b>{lvl}</b></td>'
            f'<td>{ev.get("threat_score",0):.0%}</td>'
            f'<td>{", ".join(ev.get("weapons_found",[]) or ["-"])}</td>'
            f'<td>{"Yes" if ev.get("active_grip") else "No"}</td>'
            f'</tr>'
        )
    table_html = """
    <table style="width:100%;font-size:0.75rem;border-collapse:collapse;">
      <thead>
        <tr style="color:#64748b;border-bottom:1px solid #1e2436;">
          <th align="left">Time</th><th align="left">Level</th>
          <th align="left">Score</th><th align="left">Weapons</th>
          <th align="left">Grip</th>
        </tr>
      </thead>
      <tbody>
    """ + "\n".join(rows) + "</tbody></table>"
    log_ph.markdown(table_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PROCESS LOOP  (runs once per Streamlit script re-run)
# ══════════════════════════════════════════════════════════════════

engine = get_engine()

# ── Webcam path ─────────────────────────────────────────────────
if use_webcam and frame_source is not None:
    raw_bytes = frame_source.getvalue()
    frame     = decode_frame(raw_bytes)

    if frame is not None:
        result = engine.process_frame(frame)
        fusion = result.fusion

        annotated_placeholder.image(
            bgr_to_rgb(result.annotated_frame),
            caption=f"Frame {st.session_state['frame_count']} · {result.timestamp}",
            use_container_width=True,
        )
        fps_placeholder.caption(f"FPS (moving avg): {result.fps:.1f}")

        render_dashboard(fusion, result.fps)

        if result.log_entries:
            st.session_state["alert_log"].extend(result.log_entries)

        render_log(st.session_state["alert_log"])
        st.session_state["frame_count"] += 1

# ── Uploaded video path ─────────────────────────────────────────
elif not use_webcam and uploaded_video is not None:
    # Write to temp file for OpenCV
    tmp = Path("logs/_tmp_video.mp4")
    tmp.write_bytes(uploaded_video.read())

    cap = cv2.VideoCapture(str(tmp))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.info(f"Video: {uploaded_video.name} · {total_frames} frames detected. "
            f"Processing every 3rd frame for speed.")

    progress = st.progress(0)
    stop_btn = st.button("⏹ Stop video processing")

    frame_idx = 0
    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % 3 != 0:       # process every 3rd frame
            continue

        result = engine.process_frame(frame)
        fusion = result.fusion

        annotated_placeholder.image(
            bgr_to_rgb(result.annotated_frame),
            caption=f"Frame {frame_idx}/{total_frames}",
            use_container_width=True,
        )
        fps_placeholder.caption(f"Processing FPS: {result.fps:.1f}")

        render_dashboard(fusion, result.fps)

        if result.log_entries:
            st.session_state["alert_log"].extend(result.log_entries)
        render_log(st.session_state["alert_log"])

        if total_frames > 0:
            progress.progress(min(frame_idx / total_frames, 1.0))

        time.sleep(0.01)   # yield to Streamlit renderer

    cap.release()
    progress.progress(1.0)
    st.success("Video processing complete.")

# ── Idle state ─────────────────────────────────────────────────
else:
    with col_feed:
        if use_webcam:
            st.info("📷 Click **Take photo** above (or allow camera) to start detection.")
        else:
            st.info("📂 Upload a video file in the sidebar to begin.")

    # Still render log if we have history
    render_log(st.session_state["alert_log"])

    # Default badge
    badge_ph.markdown(
        '<span class="threat-badge badge-LOW">STANDBY</span>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════

st.divider()
st.markdown(
    '<p style="font-size:0.7rem;color:#334155;text-align:center;">'
    'Parallel Threat Detection · YOLOv8 + MediaPipe · '
    'Face Analyser · Pose Analyser · Hand-Weapon Analyser · '
    'All three run in background daemon threads'
    '</p>',
    unsafe_allow_html=True,
)