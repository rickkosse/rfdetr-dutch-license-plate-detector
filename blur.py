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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# COCO class IDs (YOLO)
PERSON_CLS  = {0}
VEHICLE_CLS = {1, 2, 3, 5, 7}   # bicycle, car, motorcycle, bus, truck

# Dutch license plate sidecodes (without dashes)
_NL_PLATE_RE = re.compile(
    r'^[A-Z]{2}\d{2}[A-Z]{2}$'
    r'^[A-Z]{2}\d{3}[A-Z]{1}$'
    r'|^\d{2}[A-Z]{2}\d{2}$'
    r'|^[A-Z]{4}\d{2}$'
    r'|^\d{4}[A-Z]{2}$'
    r'|^\d{2}[A-Z]{4}$'
    r'|^[A-Z]{2}\d{4}$'
    r'|^[A-Z]{2}\d{3}[A-Z]$'
    r'|^[A-Z]\d{3}[A-Z]{2}$'
    r'|^[A-Z]{3}\d{2}[A-Z]$'
    r'|^[A-Z]\d{2}[A-Z]{3}$'
    r'|^[A-Z]{2}\d{2}[A-Z]{3}$'
    r'|^CD\d{1,4}$'
    r'|^\d{2}[A-Z]{2}\d$'
    r'|^[A-Z]\d{2}[A-Z]{2}$',
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
        x1, y1, x2, y2 = dets[i]
        x1, y1, x2, y2 = int(x1*ow), int(y1*oh), int(x2*ow), int(y2*oh)
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
                                    confidence: float = 0.4) -> tuple[list, list]:
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
        entry = {"bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                 "confidence": round(conf, 3)}
        if cls in PERSON_CLS:
            persons.append(entry)
        elif cls in VEHICLE_CLS:
            vehicles.append(entry)

    return persons, vehicles


def detect_persons_vehicles(detector, img_bgr: np.ndarray,
                             confidence: float, use_rfdetr: bool) -> tuple[list, list]:
    if use_rfdetr:
        return detect_persons_vehicles_rfdetr(detector, img_bgr, confidence)
    return detect_persons_vehicles_yolo(detector, img_bgr, confidence)


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

def process_image(img_bgr: np.ndarray, detector, plate_session, ocr,
                  exempt_plate: str | None,
                  confidence_person: float,
                  confidence_plate: float,
                  use_rfdetr: bool = False) -> tuple[np.ndarray, dict]:

    persons, vehicles = detect_persons_vehicles(detector, img_bgr, confidence_person, use_rfdetr)
    plate_dets = detect_plates(plate_session, img_bgr, confidence_plate)

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

    # Find exempt vehicle (the one whose plate matches --exempt-plate)
    exempt_vehicle = None
    if exempt_plate:
        target = re.sub(r'[\-\s]', '', exempt_plate).upper()
        for vehicle in vehicles:
            for p in vehicle["plates"]:
                if re.sub(r'[\-\s]', '', p).upper() == target:
                    exempt_vehicle = vehicle
                    break

    # Blur persons — skip those overlapping significantly with the exempt vehicle
    result = img_bgr.copy()
    blurred_count = 0
    for person in persons:
        if exempt_vehicle and overlap_fraction(person["bbox"], exempt_vehicle["bbox"]) > 0.3:
            continue
        result = blur_region(result, person["bbox"])
        blurred_count += 1

    info = {
        "persons_detected":   len(persons),
        "persons_blurred":    blurred_count,
        "vehicles_detected":  len(vehicles),
        "plates_found":       [d["plate"] for d in plate_dets if d["plate"]],
        "exempt_plate":       exempt_plate,
        "exempt_vehicle_found": exempt_vehicle is not None,
    }
    return result, info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {"*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"}
    return sorted(p for ext in exts for p in path.glob(ext))


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
    parser.add_argument("--confidence-person", type=float, default=0.4,
                        help="Person/vehicle confidence threshold (default: 0.4)")
    parser.add_argument("--confidence-plate",  type=float, default=0.3,
                        help="Plate detector confidence (default: 0.3)")
    parser.add_argument("--plate-model",     default=None,
                        help="Path to local plate ONNX model (skips HF download)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_rfdetr = args.detector == "rfdetr"

    print("Loading models...")
    model_path    = Path(args.plate_model) if args.plate_model else get_plate_model_path()
    plate_session = load_plate_session(model_path)
    detector      = load_rfdetr(args.rfdetr_large) if use_rfdetr else load_yolo()
    ocr           = load_ocr()
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

        result_img, info = process_image(
            img, detector, plate_session, ocr,
            exempt_plate=args.exempt_plate,
            confidence_person=args.confidence_person,
            confidence_plate=args.confidence_plate,
            use_rfdetr=use_rfdetr,
        )

        plates_str = ", ".join(info["plates_found"]) or "none"
        exempt_str = " (driver exempt)" if info["exempt_vehicle_found"] else ""
        print(f"  {img_path.name}: "
              f"{info['persons_blurred']}/{info['persons_detected']} persons blurred  "
              f"plates=[{plates_str}]{exempt_str}")

        cv2.imwrite(str(output_dir / img_path.name), result_img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        all_results.append({"file": img_path.name, **info})

    (output_dir / "blur_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nBlurred images: {output_dir}")


if __name__ == "__main__":
    main()
