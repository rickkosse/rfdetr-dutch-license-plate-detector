"""
detect.py — Dutch License Plate Detector
=========================================
Detects and reads Dutch license plates in images using RF-DETR + fast-plate-ocr.
The ONNX model is downloaded automatically from HuggingFace on first run.

Usage:
    python detect.py --images ./photos
    python detect.py --images ./photos --output ./results --confidence 0.4
    python detect.py --images photo.jpg                   # single file
    python detect.py --images ./photos --no-visualize     # JSON only
"""

import argparse
import json
import re
import time
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

# All valid Dutch license plate formats (sidecodes 1-10 + special categories)
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
# Model download
# ---------------------------------------------------------------------------

def get_model_path() -> Path:
    """Return local ONNX path, downloading from HuggingFace if needed."""
    local = MODEL_CACHE / ONNX_FILE
    if local.exists():
        return local

    print(f"Downloading model from HuggingFace ({HF_REPO_ID})...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("Install huggingface_hub: pip install huggingface_hub")

    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=ONNX_FILE,
        local_dir=str(MODEL_CACHE),
    )
    print(f"Model saved to {path}\n")
    return Path(path)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_session(model_path: Path):
    import onnxruntime as ort
    providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if p in ort.get_available_providers()]
    session = ort.InferenceSession(str(model_path), providers=providers)
    print(f"ONNX provider: {session.get_providers()[0]}")
    return session


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[np.newaxis]  # NCHW


def detect(session, img_bgr: np.ndarray, confidence: float,
           debug: bool = False) -> list[dict]:
    oh, ow = img_bgr.shape[:2]
    inp = preprocess(img_bgr)
    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(None, {session.get_inputs()[0].name: inp})

    if debug:
        print(f"  [debug] outputs: {output_names}")
        for name, arr in zip(output_names, outputs):
            print(f"  [debug] {name}: shape={arr.shape}  "
                  f"min={arr.min():.3f}  max={arr.max():.3f}")

    dets   = outputs[0].squeeze()   # (300, 4) normalized xyxy
    logits = np.atleast_2d(outputs[1].squeeze())   # (300, 2) raw logits

    # Automatisch de plate-klasse kolom kiezen (hoogste max-score)
    s0 = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    s1 = 1.0 / (1.0 + np.exp(-logits[:, 1]))
    scores = s0 if s0.max() > s1.max() else s1

    if debug:
        print(f"  [debug] col0 max={s0.max():.3f}  col1 max={s1.max():.3f}  "
              f"-> using col{'0' if s0.max() > s1.max() else '1'}")
        top5 = np.argsort(scores)[::-1][:5]
        print(f"  [debug] top-5 scores: {scores[top5].round(3)}")

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

        if debug:
            ratio = w / h if h > 0 else 0
            area_frac = (w * h) / (ow * oh)
            reason = ""
            if h <= 0 or not (1.5 <= ratio <= 9.0):
                reason = f"ratio={ratio:.1f} buiten [1.5-9.0]"
            elif area_frac > 0.15:
                reason = f"area={area_frac:.3f} > 0.15"
            elif w < 20 or h < 8:
                reason = f"te klein ({w}x{h})"
            print(f"  [bbox] score={scores[i]:.3f} ({x1},{y1},{w}x{h}) "
                  f"ratio={ratio:.1f} area={area_frac:.3f}"
                  + (f" -> GEFILTERD: {reason}" if reason else " -> OK"))

        if h <= 0 or not (1.5 <= w / h <= 9.0):
            continue
        if (w * h) / (ow * oh) > 0.15:
            continue
        if w < 20 or h < 8:
            continue

        results.append({
            "bbox":       [x1, y1, w, h],
            "confidence": round(float(scores[i]), 3),
        })

    results.sort(key=lambda d: d["confidence"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def load_ocr():
    from fast_plate_ocr import LicensePlateRecognizer
    return LicensePlateRecognizer("european-plates-mobile-vit-v2-model")


def read_plate(ocr, img_bgr: np.ndarray, bbox: list,
               debug: bool = False) -> str:
    x, y, w, h = bbox
    ih, iw = img_bgr.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)
    if x2 <= x1 or y2 <= y1:
        return ""
    crop_gray = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    crop_gray = crop_gray[:, :, np.newaxis]
    result = ocr.run(crop_gray)
    if not (result and result[0].plate):
        if debug:
            print(f"  [ocr] bbox=({x},{y},{w}x{h}) -> geen resultaat van OCR")
        return ""
    text = result[0].plate.strip()
    normalized = re.sub(r'[\-\s]', '', text).upper()
    if debug:
        match = bool(_NL_PLATE_RE.match(normalized))
        print(f"  [ocr] bbox=({x},{y},{w}x{h}) -> '{text}' "
              f"(norm='{normalized}') -> {'OK' if match else 'GEEN NL-formaat'}")
    return text if _NL_PLATE_RE.match(normalized) else ""


# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------

def draw_results(img: np.ndarray, detections: list, filename: str) -> np.ndarray:
    vis = img.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        plate = det.get("plate", "")
        label = f"{plate}  {det['confidence']:.2f}" if plate else f"{det['confidence']:.2f}"
        color = (0, 200, 0) if plate else (0, 165, 255)

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(vis, (x, y - th - 8), (x + tw + 6, y), color, -1)
        cv2.putText(vis, label, (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    cv2.putText(vis, filename, (10, vis.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return vis


# ---------------------------------------------------------------------------
# Main
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
    parser = argparse.ArgumentParser(description="Dutch License Plate Detector")
    parser.add_argument("--images",       required=True,
                        help="Image file or folder")
    parser.add_argument("--output",       default="./output",
                        help="Output folder (default: ./output)")
    parser.add_argument("--confidence",   type=float, default=0.01,
                        help="Detection confidence threshold (default: 0.3)")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip saving annotated images")
    parser.add_argument("--model",        default=None,
                        help="Path to local ONNX model (skips HF download)")
    parser.add_argument("--debug",        action="store_true",
                        help="Print raw model output shapes and top scores")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model) if args.model else get_model_path()
    session    = load_session(model_path)
    ocr        = load_ocr()
    print()

    images = collect_images(Path(args.images))
    if not images:
        print(f"No images found in: {args.images}")
        return

    print(f"Processing {len(images)} image(s)...\n")

    all_results = []
    t0 = time.time()

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] {img_path.name} — could not read")
            continue

        detections = detect(session, img, args.confidence, debug=args.debug)

        for det in detections:
            det["plate"] = read_plate(ocr, img, det["bbox"], debug=args.debug)

        # Remove detections where OCR found no valid plate
        detections = [d for d in detections if d["plate"]]

        n = len(detections)
        if n:
            plates = ", ".join(d["plate"] for d in detections)
            print(f"  +  {img_path.name}: {plates}")
        else:
            print(f"  -  {img_path.name}: no plates found")

        if not args.no_visualize:
            vis = draw_results(img, detections, img_path.name)
            cv2.imwrite(str(output_dir / img_path.name), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

        all_results.append({
            "file":       img_path.name,
            "detections": detections,
        })

    elapsed = time.time() - t0
    n_found = sum(1 for r in all_results if r["detections"])

    print(f"\n{'-'*45}")
    print(f"Processed : {len(all_results)} images in {elapsed:.1f}s")
    print(f"With plate: {n_found}  ({n_found / max(len(all_results), 1) * 100:.0f}%)")
    print(f"Avg speed : {elapsed / max(len(all_results), 1) * 1000:.0f} ms/image")

    json_path = output_dir / "results.json"
    json_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults   : {json_path}")
    if not args.no_visualize:
        print(f"Images    : {output_dir}")


if __name__ == "__main__":
    main()
