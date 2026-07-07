

# 🛡️ Argus — Real-Time Parallel Threat Detection System

Argus is a real-time, multi-modal AI-based surveillance system designed to detect and assess potential security threats in video feeds. The system runs multiple AI detection pipelines in parallel and fuses their outputs into a single, reliable threat assessment — reducing false alarms compared to a single-signal detector.


---

## 📌 Problem Statement

Traditional CCTV surveillance systems rely heavily on manual monitoring, which is:

- Error-prone
- Not scalable
- Slow to react in high-risk situations
- Prone to missing subtle threat indicators a human eye can overlook

Argus addresses this by combining computer vision and deep learning to automatically detect weapons and suspicious activity in real time, while cross-checking multiple signals before raising an alert.

---

## 🎯 Primary Use Cases

- Airports
- Railway & Metro Stations
- Crowded Public Places
- Offices, Campuses & Gated Communities
- Retail Stores & Banks

---

## 🧠 System Architecture Overview

Argus follows a real-time video analytics pipeline:

**1. Video Input**
Live camera feed, frame-by-frame.

**2. Weapon Detection**
A fine-tuned deep learning model scans each frame to detect guns, rifles, and knives, along with confidence scores.

**3. Parallel AI Processing Layer**
Three additional AI modules run simultaneously in the background on every frame:

- **Face Visibility Analysis** — detects whether faces are masked or covered.
- **Human Behavior Analysis** — detects suspicious posture and movement, such as raised arms, crouching, or sudden rapid motion.
- **Hand–Weapon Interaction Analysis** — checks whether a hand is actually gripping a detected weapon, confirming an active threat rather than a weapon just being present in frame.

**4. Decision & Fusion Layer**
Combines the outputs of all four modules and calculates one overall threat score, classified as Low, Medium, or High.

**5. Alert & Output Generation**
- Live threat-level display on screen
- Bounding boxes around detected weapons and hands
- Automatic screenshot capture for Medium/High threats

**6. Logging & Storage**
- Screenshot logging
- Structured event records for later review and audit

---

## 🚀 Key Features

- ⚡ Real-time processing on live video
- 🧩 Parallel AI module execution — all detectors run simultaneously, not one after another
- 🎯 Reduced false positives using multi-signal fusion and hand-grip verification
- 📊 Interpretable outputs — bounding boxes, live score breakdown, structured logs
- 🛠️ Modular, extensible architecture — each detection module works independently

---

## 🔮 Future Extensions

- **Temporal Smoothing** — analyzing several frames together instead of just one, to avoid single-frame false alarms
- **Object Tracking** — following detected weapons/people across frames rather than treating each frame as new
- **Improved Mask Detection** — replacing the current visual heuristic with a properly trained classifier
- **Depth-Aware Grip Detection** — using finger-level positioning instead of general overlap for more precise threat confirmation
- **Data-Driven Threshold Tuning** — calibrating sensitivity settings using labeled real-world data
- **Crowd & Panic Detection Module** — as a future parallel AI module for crowd density and stampede-risk prediction

---

## 🧰 Tech Stack

- **Programming Language:** Python
- **Computer Vision:** OpenCV
- **Object Detection:** YOLOv8 (fine-tuned on a custom weapon dataset via Roboflow)
- **Human Landmark Tracking:** MediaPipe (Face, Pose, Hand)
- **Deployment Ready:** Real-time camera / video stream input

---

## 🏆 Why This Approach Works

Most weapon-detection systems stop at "object detected → alarm." Argus goes further by requiring corroborating evidence before treating something as a genuine threat — a weapon on a table is treated very differently from a weapon actively gripped by someone with a covered face and aggressive posture. This layered, multi-signal approach is what keeps the system both sensitive to real threats and resistant to false alarms.
