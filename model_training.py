"""
finetune_yolov8.py
══════════════════════════════════════════════════════════════════
Fine-tunes a pretrained YOLOv8 model on your already-downloaded
Roboflow multi-weapon dataset.

Make sure your dataset folder looks like this (standard Roboflow export):

    weapon-dataset/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
══════════════════════════════════════════════════════════════════
"""

from ultralytics import YOLO

# ── 1. Load a pretrained YOLOv8 model ────────────────────────────
model = YOLO("yolov8n.pt")   # pretrained on COCO; use yolov8s.pt/yolov8m.pt for more accuracy

# ── 2. Fine-tune it on your weapon dataset ───────────────────────
results = model.train(
    data="weapon-dataset/data.yaml",   # path to the data.yaml from your downloaded dataset
    epochs=50,
    imgsz=640,
    batch=16,
    device="0",        # use "0" for GPU, or "cpu" if no GPU
    name="weapon_detector"
)

# ── 3. Validate the fine-tuned model ─────────────────────────────
metrics = model.val()
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision:{metrics.box.mp:.4f}")
print(f"Recall:   {metrics.box.mr:.4f}")

# ── 4. Save the fine-tuned model as yolov8_updated.pt ────────────
import shutil
best_weights = "runs/detect/weapon_detector/weights/best.pt"
shutil.copy(best_weights, "yolov8_updated.pt")

print("Fine-tuning complete. Model saved as yolov8_updated.pt")