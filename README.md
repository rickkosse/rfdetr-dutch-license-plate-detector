# Dutch License Plate Detector

Detects and reads Dutch license plates using **RF-DETR** + **fast-plate-ocr**.  
The ONNX model downloads automatically from HuggingFace on first run — no manual setup needed.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Single image
python detect.py --images photo.jpg

# Folder of images
python detect.py --images ./photos

# Custom output folder and confidence
python detect.py --images ./photos --output ./results --confidence 0.4

# JSON only, no annotated images
python detect.py --images ./photos --no-visualize

# Use a local ONNX model (skip HF download)
python detect.py --images ./photos --model ./inference_model.onnx
```

## Output

Annotated images and a `results.json` are saved to `./output/` (or `--output`).

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

Terminal output:

```
Processing 3 image(s)...

  ✓  photo1.jpg: 52WDT9
  ✓  photo2.jpg: JKB18V, 33STXH
  -  photo3.jpg: no plates found

─────────────────────────────────────────────
Processed : 3 images in 4.1s
With plate: 2  (67%)
Avg speed : 1367 ms/image
```

## How it works

1. **Detection** — RF-DETR Base (ONNX, 560×560) finds plate bounding boxes
2. **Geometry filter** — rejects boxes with wrong aspect ratio or area
3. **OCR** — `fast-plate-ocr` (`european-plates-mobile-vit-v2-model`) reads the text
4. **Format validation** — regex check against all 10 Dutch sidecodes + special categories

Supported plate formats: sidecodes 1–10, agricultural (sidecode 11), diplomatic (CD), mopeds.

## Model

Hosted on HuggingFace: [Rickkosse/rfdetr-flitsfoto-detector](https://huggingface.co/Rickkosse/rfdetr-flitsfoto-detector)

- Architecture: RF-DETR Base, 1 class (`license_plate`)
- Resolution: 560×560
- Training: synthetic plates on BDD100K + real-world crops
- EMA checkpoint, cosine LR schedule
