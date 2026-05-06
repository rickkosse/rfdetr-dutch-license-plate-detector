"""
blur.py — Privacy Blurring Pipeline
=====================================
Detects persons, vehicles and license plates in images.
All persons are blurred, except those associated with a specified exempt plate.

Dependencies:
    pip install ultralytics onnxruntime fast-plate-ocr opencv-python huggingface_hub rfdetr

Usage:
    # Blur all persons (YOLO, default)
    python blur.py --images ./photos

    # Use RF-DETR instead of YOLO for person/vehicle detection
    python blur.py --images ./photos --detector rfdetr
    python blur.py --images ./photos --detector rfdetr --rfdetr-large

    # Exempt the driver of a specific plate from blurring
    python blur.py --images ./photos --exempt-plate 52WDT9

    # Single image, custom output folder
    python blur.py --images photo.jpg --output ./blurred --exempt-plate JKB18V
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_REPO_ID   = "Rickkosse/rfdetr_licences_plate_detector"
ONNX_FILE    = "inference_model.onnx"
MODEL_CACHE  = Path.home() / ".cache" / "dutch-plate-detector"
INPUT_SIZE   = 560

FACE_PROTOTXT_URL   = ("https://raw.githubusercontent.com/opencv/opencv/4.x"
                        "/samples/dnn/face_detector/deploy.prototxt")
FACE_MODEL_URL      = ("https://github.com/opencv/opencv_3rdparty/raw/"
                        "dnn_samples_face_detector_20170830/"
                        "res10_300x300_ssd_iter_140000.caffemodel")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# COCO class IDs (YOLO)
PERSON_CLS  = {0}
VEHICLE_CLS = {1, 2, 3, 5, 7}   # bicycle, car, motorcycle, bus, truck

# Dutch license plate sidecodes (without dashes)
_NL_PLATE_RE = re.compile(
    r'^[A-Z]{2}\d{2}[A-Z]{2}$'   # sidecode 1:  XX-99-XX
    r'|^\d{2}[A-Z]{2}\d{2}$'     # sidecode 2:  99-XX-99
    r'|^[A-Z]{4}\d{2}$'          # sidecode 3:  XX-XX-99
    r'|^\d{4}[A-Z]{2}$'          # sidecode 4:  99-99-XX
    r'|^\d{2}[A-Z]{4}$'          # sidecode 5:  99-XX-XX
    r'|^[A-Z]{2}\d{4}$'          # sidecode 6:  XX-99-99
    r'|^[A-Z]{2}\d{3}[A-Z]$'     # sidecode 7:  XX-999-X
    r'|^[A-Z]\d{3}[A-Z]{2}$'     # sidecode 8:  X-999-XX
    r'|^[A-Z]{3}\d{2}[A-Z]$'     # sidecode 9:  XXX-99-X
    r'|^[A-Z]\d{2}[A-Z]{3}$'     # sidecode 10: X-99-XXX
    r'|^\d{2}[A-Z]{3}\d$'        # sidecode 12: 99-XXX-9
    r'|^\d[A-Z]{3}\d{2}$'        # sidecode 13: 9-XXX-99
    r'|^[A-Z]{2}\d{2}[A-Z]{3}$'  # sidecode 14: XX-99-XXX (agricultural)
    r'|^CD\d{1,4}$'              # Diplomatic
    r'|^\d{2}[A-Z]{2}\d$'        # Moped
    r'|^[A-Z]\d{2}[A-Z]{2}$',    # Light moped
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_plate_model_path() -> Path:
    local = MODEL_CACHE / ONNX_FILE
    if local.exists():
        return local
    print(f"Downloading plate model from HuggingFace ({HF_REPO_ID})...")
    from huggingface_hub import hf_hub_download
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=ONNX_FILE,
                           local_dir=str(MODEL_CACHE))
    print(f"Saved to {path}\n")
    return Path(path)


def load_plate_session(model_path: Path):
    import onnxruntime as ort
    providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if p in ort.get_available_providers()]
    session = ort.InferenceSession(str(model_path), providers=providers)
    print(f"Plate model: ONNX ({session.get_providers()[0]})")
    return session


def load_yolo():
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")   # downloads ~6 MB automatically on first run
    print("Person/vehicle model: YOLO11n")
    return model


def load_rfdetr(large: bool = False):
    import warnings
    from rfdetr import RFDETRBase, RFDETRLarge
    cls  = RFDETRLarge if large else RFDETRBase
    name = "RFDETRLarge" if large else "RFDETRBase"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = cls()   # downloads COCO pretrained weights automatically
    print(f"Person/vehicle model: {name} (COCO)")
    return model


def load_ocr():
    from fast_plate_ocr import LicensePlateRecognizer
    ocr = LicensePlateRecognizer("european-plates-mobile-vit-v2-model")
    print("OCR model: european-plates-mobile-vit-v2")
    return ocr


def load_face_detector():
    import urllib.request
    face_cache = MODEL_CACHE / "face_detector"
    face_cache.mkdir(parents=True, exist_ok=True)
    prototxt   = face_cache / "deploy.prototxt"
    caffemodel = face_cache / "res10_300x300.caffemodel"
    if not prototxt.exists():
        print("Downloading face detector config...")
        urllib.request.urlretrieve(FACE_PROTOTXT_URL, prototxt)
    if not caffemodel.exists():
        print("Downloading face detector model (~10 MB)...")
        urllib.request.urlretrieve(FACE_MODEL_URL, caffemodel)
    net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
    print("Face fallback: OpenCV DNN SSD ResNet10")
    return net


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def preprocess_plate(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[np.newaxis]


def detect_plates(session, img_bgr: np.ndarray,
                  confidence: float = 0.3) -> list[dict]:
    oh, ow = img_bgr.shape[:2]
    inp = preprocess_plate(img_bgr)
    outputs = session.run(None, {session.get_inputs()[0].name: inp})

    dets   = outputs[0].squeeze()
    logits = np.atleast_2d(outputs[1].squeeze())
    s0 = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    s1 = 1.0 / (1.0 + np.exp(-logits[:, 1]))
    scores = s0 if s0.max() > s1.max() else s1

    results = []
    for i in np.where(scores > confidence)[0]:
        cx, cy, bw, bh = dets[i]   # cxcywh genormaliseerd
        x1 = int((cx - bw / 2) * ow)
        y1 = int((cy - bh / 2) * oh)
        x2 = int((cx + bw / 2) * ow)
        y2 = int((cy + bh / 2) * oh)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ow, x2), min(oh, y2)
        w, h = x2 - x1, y2 - y1
        if h <= 0 or not (1.5 <= w / h <= 9.0):
            continue
        if (w * h) / (ow * oh) > 0.15 or w < 20 or h < 8:
            continue
        results.append({"bbox": [x1, y1, w, h], "confidence": round(float(scores[i]), 3)})

    return sorted(results, key=lambda d: d["confidence"], reverse=True)


def read_plate(ocr, img_bgr: np.ndarray, bbox: list) -> str:
    x, y, w, h = bbox
    ih, iw = img_bgr.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(iw, x+w), min(ih, y+h)
    if x2 <= x1 or y2 <= y1:
        return ""
    crop_gray = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    result = ocr.run(crop_gray[:, :, np.newaxis])
    if not (result and result[0].plate):
        return ""
    text = result[0].plate.strip()
    return text if _NL_PLATE_RE.match(re.sub(r'[\-\s]', '', text).upper()) else ""


def detect_persons_vehicles_yolo(yolo, img_bgr: np.ndarray,
                                  confidence: float = 0.4) -> tuple[list, list]:
    results = yolo(img_bgr, conf=confidence, verbose=False)[0]
    persons, vehicles = [], []
    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        entry = {"bbox": [x1, y1, x2-x1, y2-y1], "confidence": round(conf, 3)}
        if cls in PERSON_CLS:
            persons.append(entry)
        elif cls in VEHICLE_CLS:
            vehicles.append(entry)
    return persons, vehicles


def detect_persons_vehicles_rfdetr(model, img_bgr: np.ndarray,
                                    confidence: float = 0.4,
                                    debug: bool = False) -> tuple[list, list]:
    import PIL.Image
    import warnings
    pil_img = PIL.Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model.predict(pil_img, threshold=confidence)

    persons, vehicles = [], []
    if not hasattr(results, "xyxy") or len(results.xyxy) == 0:
        return persons, vehicles

    for i in range(len(results.xyxy)):
        x1, y1, x2, y2 = results.xyxy[i].astype(int)
        conf = float(results.confidence[i])
        if conf < confidence:
            continue
        # class_id attribute name varies by rfdetr version
        cls = int(
            results.class_id[i] if hasattr(results, "class_id")
            else results.labels[i] if hasattr(results, "labels")
            else -1
        )
        # RF-DETR may return 1-indexed COCO IDs — normalise to 0-indexed
        cls0 = cls - 1 if cls >= 1 else cls

        if debug:
            label = getattr(results, "class_names", {}).get(cls, f"cls{cls}")
            print(f"    [rfdetr-raw] cls={cls} (0idx={cls0}) conf={conf:.2f} "
                  f"label={label} bbox=[{x1},{y1},{x2-x1},{y2-y1}]")

        entry = {"bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                 "confidence": round(conf, 3)}
        # Try 0-indexed first, fall back to 1-indexed if nothing matches
        if cls0 in PERSON_CLS or cls in PERSON_CLS:
            persons.append(entry)
        elif cls0 in VEHICLE_CLS or cls in VEHICLE_CLS:
            vehicles.append(entry)

    return persons, vehicles


def detect_persons_vehicles(detector, img_bgr: np.ndarray,
                             confidence: float, use_rfdetr: bool,
                             debug: bool = False) -> tuple[list, list]:
    if use_rfdetr:
        return detect_persons_vehicles_rfdetr(detector, img_bgr, confidence, debug=debug)
    return detect_persons_vehicles_yolo(detector, img_bgr, confidence)


def detect_faces(net, img_bgr: np.ndarray,
                 offset_x: int = 0, offset_y: int = 0,
                 confidence: float = 0.5,
                 debug: bool = False,
                 debug_label: str = "") -> list[dict]:
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0,
                                  (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    dets = net.forward()
    out = []

    if debug:
        scores = sorted([float(dets[0, 0, i, 2]) for i in range(dets.shape[2])], reverse=True)
        top3 = [f"{s:.3f}" for s in scores[:3]]
        print(f"      [face-dnn] {debug_label} ({w}x{h}) top3={top3}")

    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < confidence:
            continue
        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)
        fw, fh = x2 - x1, y2 - y1
        if fw < 10 or fh < 10:
            continue
        # Reject detections that cover more than 60% of the crop — clearly not a face
        if fw > w * 0.6 or fh > h * 0.6:
            if debug:
                print(f"      [face-dnn]   -> rejected oversized {fw}x{fh} in {w}x{h}")
            continue
        pad = int(fh * 0.4)
        out.append({
            "bbox": [offset_x + x1 - pad, offset_y + y1 - pad,
                     fw + 2 * pad, fh + 2 * pad],
            "confidence": round(conf, 3),
            "source": "face",
        })
    return out


def detect_faces_multiscale(net, img_bgr: np.ndarray,
                             offset_x: int = 0, offset_y: int = 0,
                             confidence: float = 0.35,
                             debug: bool = False,
                             debug_label: str = "") -> list[dict]:
    """Run face DNN on full crop + horizontal strips + quadrants.

    Large vehicle bboxes shrink the face to a tiny fraction of the 300x300
    DNN input. Running on sub-regions keeps the face large enough to detect.
    """
    h, w = img_bgr.shape[:2]
    regions: list[tuple[int, int, int, int]] = [(0, 0, w, h)]   # full crop

    # Horizontal halves and thirds (catches cab vs cargo split)
    if h >= 80:
        mid = h // 2
        regions += [(0, 0, w, mid), (0, mid, w, h - mid)]
        if h >= 120:
            t = h // 3
            regions += [(0, 0, w, t), (0, t, w, t), (0, 2*t, w, h - 2*t)]

    # Quadrants for wide crops
    if w >= 80 and h >= 80:
        mw, mh = w // 2, h // 2
        regions += [
            (0,  0,  mw, mh), (mw, 0,  w - mw, mh),
            (0,  mh, mw, h - mh), (mw, mh, w - mw, h - mh),
        ]

    faces: list[dict] = []
    for idx, (rx, ry, rw, rh) in enumerate(regions):
        if rw < 30 or rh < 30:
            continue
        crop = img_bgr[ry:ry + rh, rx:rx + rw]
        if crop.size == 0:
            continue
        region_label = f"{debug_label}[r{idx} +{rx},{ry}]" if debug_label else f"r{idx} +{rx},{ry}"
        for f in detect_faces(net, crop, offset_x + rx, offset_y + ry, confidence,
                              debug=debug, debug_label=region_label):
            if not any(overlap_fraction(f["bbox"], e["bbox"]) > 0.5 for e in faces):
                faces.append(f)
    return faces


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def bbox_to_xyxy(bbox: list) -> tuple:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = bbox_to_xyxy(a)
    bx1, by1, bx2, by2 = bbox_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2-ax1) * (ay2-ay1)
    area_b = (bx2-bx1) * (by2-by1)
    return inter / (area_a + area_b - inter)


def overlap_fraction(inner: list, outer: list) -> float:
    """Fraction of `inner` bbox that falls inside `outer` bbox."""
    ax1, ay1, ax2, ay2 = bbox_to_xyxy(inner)
    bx1, by1, bx2, by2 = bbox_to_xyxy(outer)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area_inner = (ax2-ax1) * (ay2-ay1)
    return inter / area_inner if area_inner > 0 else 0.0


# ---------------------------------------------------------------------------
# Blurring
# ---------------------------------------------------------------------------

def windshield_region(vehicle_bbox: list) -> list:
    """Upper 55% of vehicle bbox — covers windshield from any camera angle."""
    x, y, w, h = vehicle_bbox
    margin = int(w * 0.08)
    return [x + margin, y, w - 2 * margin, int(h * 0.55)]


def blur_region(img: np.ndarray, bbox: list, strength: int = 51) -> np.ndarray:
    x, y, w, h = bbox
    ih, iw = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(iw, x+w), min(ih, y+h)
    if x2 <= x1 or y2 <= y1:
        return img
    k = strength | 1   # must be odd
    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return img


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def draw_debug(img_bgr: np.ndarray, persons: list, vehicles: list,
               plate_dets: list) -> np.ndarray:
    vis = img_bgr.copy()
    for v in vehicles:
        x, y, w, h = v["bbox"]
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 100, 0), 2)
        cv2.putText(vis, f"vehicle {v['confidence']:.2f}", (x, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
    for p in plate_dets:
        x, y, w, h = p["bbox"]
        label = p.get("plate") or f"{p['confidence']:.2f}"
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 200, 0), 2)
        cv2.putText(vis, label, (x, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    for p in persons:
        x, y, w, h = p["bbox"]
        src = p.get("source", "yolo")
        if src == "windshield":
            color, label = (0, 165, 255), "windshield"
        elif src == "face":
            color, label = (0, 200, 255), f"face {p['confidence']:.2f}"
        else:
            color, label = (0, 0, 255), f"person {p['confidence']:.2f}"
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        cv2.putText(vis, label, (x, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


def process_image(img_bgr: np.ndarray, detector, plate_session, ocr,
                  face_detector,
                  exempt_plate: str | None,
                  confidence_person: float,
                  confidence_plate: float,
                  use_rfdetr: bool = False,
                  windshield_fallback: bool = False,
                  debug: bool = False) -> tuple[np.ndarray, dict]:

    persons, vehicles = detect_persons_vehicles(detector, img_bgr, confidence_person, use_rfdetr, debug=debug)
    plate_dets = detect_plates(plate_session, img_bgr, confidence_plate)

    if debug:
        n_yolo_persons  = len(persons)
        n_yolo_vehicles = len(vehicles)
        print(f"    [debug] YOLO/RF-DETR: {n_yolo_persons} person(s), {n_yolo_vehicles} vehicle(s), "
              f"{len(plate_dets)} plate bbox(es)")
        for v in vehicles:
            print(f"    [debug]   vehicle conf={v['confidence']:.2f} bbox={v['bbox']}")
        for p in persons:
            print(f"    [debug]   person  conf={p['confidence']:.2f} bbox={p['bbox']}")

    # Read OCR for every detected plate bbox
    for det in plate_dets:
        det["plate"] = read_plate(ocr, img_bgr, det["bbox"])

    # Associate each plate with the vehicle bbox that contains it most
    for vehicle in vehicles:
        vehicle["plates"] = []
    for det in plate_dets:
        if not det["plate"]:
            continue
        best_vehicle, best_frac = None, 0.0
        for vehicle in vehicles:
            frac = overlap_fraction(det["bbox"], vehicle["bbox"])
            if frac > best_frac:
                best_frac, best_vehicle = frac, vehicle
        if best_vehicle is not None:
            best_vehicle["plates"].append(det["plate"])

    # Resolve exempt vehicle now so face detection can skip it
    exempt_vehicle = None
    exempt_normalized = None
    if exempt_plate:
        exempt_normalized = re.sub(r'[\-\s]', '', exempt_plate).upper()
        for vehicle in vehicles:
            for p in vehicle["plates"]:
                if re.sub(r'[\-\s]', '', p).upper() == exempt_normalized:
                    exempt_vehicle = vehicle
                    break

    # Face detection fallback
    if face_detector is not None:
        ih, iw = img_bgr.shape[:2]

        # Pass 1: face DNN on every vehicle crop except the exempt vehicle.
        # Runs regardless of whether YOLO already found a person (catches passengers).
        vehicles_without_person = []
        for vehicle in vehicles:
            if vehicle is exempt_vehicle:
                continue
            vx, vy, vw, vh = vehicle["bbox"]
            x1, y1 = max(0, vx), max(0, vy)
            x2, y2 = min(iw, vx + vw), min(ih, vy + vh)
            crop = img_bgr[y1:y2, x1:x2]
            if (x2 - x1) >= 40 and (y2 - y1) >= 40 and crop.size > 0:
                if debug:
                    print(f"    [debug]   Pass1 face DNN on vehicle {vehicle['bbox']}:")
                found = detect_faces_multiscale(face_detector, crop,
                                               offset_x=x1, offset_y=y1,
                                               debug=debug,
                                               debug_label=f"veh{vehicle['bbox']}")
                # Deduplicate against already-found persons
                for f in found:
                    if not any(overlap_fraction(f["bbox"], p["bbox"]) > 0.5
                               for p in persons):
                        persons.append(f)
                if debug:
                    print(f"    [debug]   -> {len(found)} face(s) above threshold")
            has_person = any(overlap_fraction(p["bbox"], vehicle["bbox"]) > 0.3
                             for p in persons)
            if not has_person:
                vehicles_without_person.append(vehicle)

        # Pass 2: face DNN on full image to catch pedestrians outside vehicle bboxes
        if debug:
            print(f"    [debug]   Pass2 face DNN on full image:")
        full_faces = detect_faces_multiscale(face_detector, img_bgr,
                                             debug=debug, debug_label="full")
        for f in full_faces:
            inside_any_vehicle = any(
                overlap_fraction(f["bbox"], v["bbox"]) > 0.3 for v in vehicles
            )
            already_covered = any(
                overlap_fraction(f["bbox"], p["bbox"]) > 0.5 for p in persons
            )
            if not inside_any_vehicle and not already_covered:
                if debug:
                    print(f"    [debug]   face DNN pedestrian: {f['bbox']} conf={f['confidence']}")
                persons.append(f)

        # Pass 3: windshield heuristic (opt-in via --windshield-fallback)
        if windshield_fallback:
            for vehicle in vehicles_without_person:
                if vehicle is exempt_vehicle:
                    continue
                has_person_now = any(
                    overlap_fraction(p["bbox"], vehicle["bbox"]) > 0.3 for p in persons
                )
                if not has_person_now:
                    ws = windshield_region(vehicle["bbox"])
                    if debug:
                        print(f"    [debug]   windshield heuristic for vehicle "
                              f"bbox={vehicle['bbox']} -> blur {ws}")
                    persons.append({"bbox": ws, "confidence": 1.0, "source": "windshield"})

    if debug:
        print(f"    [debug] total persons to blur: {len(persons)} "
              f"({len(persons) - n_yolo_persons} from face/windshield fallback)")

    # Blur persons — skip those overlapping with the exempt vehicle
    result = img_bgr.copy()
    blurred_count = 0
    for person in persons:
        if exempt_vehicle and overlap_fraction(person["bbox"], exempt_vehicle["bbox"]) > 0.3:
            continue
        result = blur_region(result, person["bbox"])
        blurred_count += 1

    # Blur license plates — skip only the exempt plate (if specified)
    plates_blurred = 0
    for det in plate_dets:
        if exempt_normalized and det.get("plate"):
            if re.sub(r'[\-\s]', '', det["plate"]).upper() == exempt_normalized:
                continue
        result = blur_region(result, det["bbox"])
        plates_blurred += 1

    info = {
        "persons_detected":    len(persons),
        "persons_blurred":     blurred_count,
        "vehicles_detected":   len(vehicles),
        "plates_found":        [d["plate"] for d in plate_dets if d["plate"]],
        "plates_blurred":      plates_blurred,
        "exempt_plate":        exempt_plate,
        "exempt_vehicle_found": exempt_vehicle is not None,
        "_persons":            persons,
        "_vehicles":           vehicles,
        "_plate_dets":         plate_dets,
    }
    return result, info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {"*.jpg", "*.jpeg", "*.png"}
    seen = set()
    result = []
    for ext in exts:
        for p in path.glob(ext):
            key = p.resolve()
            if key not in seen:
                seen.add(key)
                result.append(p)
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description="Privacy blurring pipeline")
    parser.add_argument("--images",          required=True,
                        help="Image file or folder")
    parser.add_argument("--output",          default="./blurred",
                        help="Output folder (default: ./blurred)")
    parser.add_argument("--exempt-plate",    default=None,
                        help="License plate whose driver should NOT be blurred")
    parser.add_argument("--detector",        default="yolo",
                        choices=["yolo", "rfdetr"],
                        help="Person/vehicle detector: yolo (default) or rfdetr")
    parser.add_argument("--rfdetr-large",    action="store_true",
                        help="Use RFDETRLarge instead of RFDETRBase (rfdetr only)")
    parser.add_argument("--confidence-person", type=float, default=0.3,
                        help="Person/vehicle confidence threshold (default: 0.3)")
    parser.add_argument("--confidence-plate",  type=float, default=0.3,
                        help="Plate detector confidence (default: 0.3)")
    parser.add_argument("--plate-model",     default=None,
                        help="Path to local plate ONNX model (skips HF download)")
    parser.add_argument("--windshield-fallback", action="store_true",
                        help="Blur upper vehicle area when no person or face is detected (conservative)")
    parser.add_argument("--debug",           action="store_true",
                        help="Print per-image detection counts and save debug images")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_rfdetr = args.detector == "rfdetr"

    print("Loading models...")
    model_path    = Path(args.plate_model) if args.plate_model else get_plate_model_path()
    plate_session = load_plate_session(model_path)
    detector      = load_rfdetr(args.rfdetr_large) if use_rfdetr else load_yolo()
    ocr           = load_ocr()
    face_detector  = load_face_detector()
    print()

    if args.exempt_plate:
        print(f"Exempt plate : {args.exempt_plate.upper()} (driver will not be blurred)\n")

    images = collect_images(Path(args.images))
    if not images:
        print(f"No images found in: {args.images}")
        return

    print(f"Processing {len(images)} image(s)...\n")

    all_results = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] {img_path.name}")
            continue

        if args.debug:
            print(f"  {img_path.name}:")

        result_img, info = process_image(
            img, detector, plate_session, ocr,
            face_detector=face_detector,
            exempt_plate=args.exempt_plate,
            confidence_person=args.confidence_person,
            confidence_plate=args.confidence_plate,
            use_rfdetr=use_rfdetr,
            windshield_fallback=args.windshield_fallback,
            debug=args.debug,
        )

        plates_str = ", ".join(info["plates_found"]) or "none"
        exempt_str = f"  exempt={args.exempt_plate}" if info["exempt_vehicle_found"] else ""
        print(f"  {img_path.name}: "
              f"{info['persons_blurred']}/{info['persons_detected']} persons blurred  "
              f"{info['plates_blurred']} plate(s) blurred  "
              f"plates=[{plates_str}]{exempt_str}")

        if args.debug:
            # Save a side-by-side debug image showing all raw detections
            debug_img = draw_debug(img, info.get("_persons", []),
                                   info.get("_vehicles", []), info.get("_plate_dets", []))
            stem = img_path.stem
            cv2.imwrite(str(output_dir / f"{stem}_debug.jpg"), debug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

        cv2.imwrite(str(output_dir / img_path.name), result_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        public_info = {k: v for k, v in info.items() if not k.startswith("_")}
        all_results.append({"file": img_path.name, **public_info})

    (output_dir / "blur_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nBlurred images: {output_dir}")


if __name__ == "__main__":
    main()
