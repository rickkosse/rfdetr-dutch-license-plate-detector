Hier is de volledige README.md met **Apache 2.0-licentie**, gebaseerd op de eerdere verbeteringen.

```markdown
# Dutch License Plate Detector & Privacy Blurrer

Two Python scripts for **Dutch license plate detection**, **OCR**, and **privacy blurring** (persons + plates).  
Models are downloaded automatically from HuggingFace and Ultralytics on first run – no manual setup required.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 📦 Installation

```bash
pip install -r requirements.txt
```

> ℹ️ The first time you run a script, required models are downloaded automatically:
> - **RF-DETR** plate detector (from HuggingFace)
> - **fast-plate-ocr** model (from HuggingFace)
> - **YOLO11n** or **RF-DETR** (person/vehicle detection, depending on your flags)

---

## 🚗 `detect.py` – Plate detection + OCR

Find and read license plates in images.

### Basic usage

```bash
# Single image
python detect.py --images photo.jpg

# Folder of images
python detect.py --images ./photos
```

### Options

| Flag | Description |
|------|-------------|
| `--images` | Path to an image or folder (required) |
| `--output` | Output directory (default: `./output`) |
| `--confidence` | Confidence threshold for plate detection (default: `0.4`) |
| `--no-visualize` | Save JSON results only, skip annotated images |
| `--model` | Path to a local ONNX model (skip HuggingFace download) |

### Example

```bash
python detect.py --images ./photos --output ./results --confidence 0.4 --no-visualize
```

### Output

- **Annotated images** (unless `--no-visualize` is used)  
- **`results.json`** – detection data in the output folder

#### JSON structure

```json
[
  {
    "file": "photo.jpg",
    "detections": [
      { "bbox": [120, 340, 180, 42], "confidence": 0.92, "plate": "52WDT9" }
    ]
  }
]
```

#### Console output example

```
Processing 3 image(s)...

  +  photo1.jpg: 52WDT9
  +  photo2.jpg: JKB18V, 33STXH
  -  photo3.jpg: no plates found

---------------------------------------------
Processed : 3 images in 4.1s
With plate: 2  (67%)
Avg speed : 1367 ms/image
```

| Single plate | Multi-vehicle |
|--------------|---------------|
| ![single plate](examples/detect_single.jpg) | ![multi vehicle](examples/detect_result.png) |

---

## 🔒 `blur.py` – Privacy blurring

Detect **persons**, **vehicles**, and **license plates** – then blur all persons and plates.  
Optionally, exempt the driver of a specific vehicle from blurring.

### Basic usage

```bash
# Blur all persons and plates (YOLO, default)
python blur.py --images ./photos

# Better person/vehicle detection (recommended)
python blur.py --images ./photos --detector rfdetr

# Exempt the driver of a specific plate
python blur.py --images ./photos --detector rfdetr --exempt-plate OL70ZL
```

### Options

| Flag | Description |
|------|-------------|
| `--images` | Path to an image or folder (required) |
| `--output` | Output directory (default: `./blurred`) |
| `--detector` | Person/vehicle detector: `yolo` (default) or `rfdetr` |
| `--rfdetr-large` | Use RF-DETR Large (more accurate, slower) |
| `--exempt-plate` | Do not blur persons associated with this plate (e.g., driver) |
| `--confidence-person` | Confidence threshold for person detection (default: `0.5`) |
| `--confidence-plate` | Confidence threshold for plate detection (default: `0.4`) |
| `--debug` | Show detection boxes; saves debug images in output folder |
| `--windshield-fallback` | For aerial/overhead shots: fallback to blurring windshield area when no face is visible |
| `--plate-model` | Path to a local plate detection ONNX model (skip HuggingFace download) |

### Detector comparison

| Detector | Speed | Accuracy | When to use |
|----------|-------|----------|-------------|
| `yolo` (default) | ⚡ Fast | ✅ Good | Standard scenes, good lighting |
| `rfdetr` | 🐢 Slower | ⭐ Higher | Partially visible persons, angled shots |
| `rfdetr --rfdetr-large` | 🐌 Slowest | 🌟 Best | Maximum accuracy, offline processing |

### Output

- **Blurred images** saved to output folder  
- **`blur_results.json`** – detailed detection and blurring statistics

#### JSON structure

```json
[
  {
    "file": "photo.jpg",
    "persons_detected": 3,
    "persons_blurred": 3,
    "vehicles_detected": 3,
    "plates_found": ["OL70ZL", "G361RX", "JN231Z"],
    "plates_blurred": 3,
    "exempt_plate": "OL70ZL",
    "exempt_vehicle_found": true
  }
]
```

#### Visual examples

| Original | Debug (boxes) | Result (blurred) |
|----------|---------------|------------------|
| ![original](examples/blur_original.png) | ![debug](examples/blur_debug.jpg) | ![result](examples/blur_result.png) |

> **Debug mode** (`--debug`) shows:  
> 🔵 Blue boxes = vehicles &nbsp;|&nbsp; 🔴 Red boxes = persons &nbsp;|&nbsp; 🟢 Green boxes = plates

---

## ⚙️ How it works

### Plate detection (`detect.py` + `blur.py`)

1. **RF-DETR Base (ONNX, 560×560)** – finds plate bounding boxes  
2. **Geometry filter** – rejects invalid boxes (aspect ratio <1.5 or >9.0, area >15% of image)  
3. **fast-plate-ocr** (`european-plates-mobile-vit-v2-model`) – reads plate text  
4. **Format validation** – regex check against all Dutch sidecodes (1–14), agricultural, diplomatic and moped formats

### Privacy blurring (`blur.py`)

1. **Person/vehicle detection** – YOLO11n or RF-DETR (COCO 80 classes)  
2. **Face fallback** – OpenCV DNN SSD ResNet10 runs multiscale on vehicle crops + full image to catch missed persons  
3. **Plate association** – each plate is linked to the vehicle bbox with highest overlap  
4. **Exempt logic** – if `--exempt-plate` is set, persons overlapping that vehicle are *not* blurred  
5. **Plate blurring** – all non‑exempt plates are blurred  
6. **Gaussian blur** – applied to all non‑exempt persons and plates

---

## 📁 Model sources

| Model | Source |
|-------|--------|
| Plate detector (RF-DETR) | [Rickkosse/rfdetr_licences_plate_detector](https://huggingface.co/Rickkosse/rfdetr_licences_plate_detector) – trained on synthetic plates + BDD100K |
| OCR | [fast-plate-ocr](https://huggingface.co/xonrix/fast-plate-ocr) – `european-plates-mobile-vit-v2-model` |
| Person/vehicle detector | Ultralytics YOLO11n or HuggingFace RF-DETR (COCO) |

---

## ❓ Troubleshooting

- **First run downloads models** – ensure you have a working internet connection.  
- **No plates found** – try lowering `--confidence` (e.g., `0.3`).  
- **Slow performance with RF-DETR** – use `--detector yolo` for speed, or add `--rfdetr-large` only if needed.  
- **Exempt driver not working** – verify that the plate is correctly detected and associated with a vehicle (use `--debug` to inspect overlaps).  

---

## 📄 License

Apache 2.0 License – free for personal and commercial use; see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [RF-DETR](https://github.com/roboflow/rfdetr) by Roboflow  
- [fast-plate-ocr](https://github.com/xonrix/fast-plate-ocr)  
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
```

**Opmerking:** Vergeet niet het `LICENSE`-bestand met de volledige Apache 2.0-tekst aan je repository toe te voegen. De badge en de verwijzing in de README gaan er dan van uit dat dat bestand bestaat.
