# Car Counting System

YOLO11s + ByteTrack + Gradio — detect, track, and count vehicles in traffic video.

## How It Works

1. **YOLO11s** detects vehicles across 7 classes
2. **ByteTrack** assigns persistent tracking IDs across frames
3. A **virtual horizontal line** counts cars crossing from top to bottom
4. Only **passenger cars** are counted (Sedan, SUV, Pickup)
5. Each track ID is counted **exactly once**

## Project Structure

| Dir/File             | Purpose                                       |
| -------------------- | --------------------------------------------- |
| `dataset/`           | Images + YOLO labels (train/valid/test)       |
| `notebooks/`         | Training + inference notebooks, standalone .py |
| `models/`            | Trained model weights (`best.pt`)              |
| `videos/`            | Input videos for testing                       |
| `outputs/`           | Processed videos (detection/tracking/counting) |

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For GPU inference (recommended), reinstall PyTorch with CUDA:

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Skip the last step for CPU-only.

## Usage

### Training

Run the training notebook from the `notebooks/` directory:

```
cd notebooks
jupyter notebook car_training.ipynb
```

The notebook handles class label cleanup, dataset verification, YOLO11s fine-tuning, and saves `models/best.pt`.

### Inference — Python Script

Run the Gradio web app from the project root:

```
.venv\Scripts\python notebooks\car_counting_gradio.py
```

Open `http://127.0.0.1:7860`, upload a traffic video, and click **Process Video**. The processed video with tracking overlays and the car count are displayed in the interface.

### Inference — Notebook

Alternative: run the notebook version from `notebooks/`:

```
cd notebooks
jupyter notebook car_counting_gradio.ipynb
```

## Dataset

- **Source**: [Roboflow](https://universe.roboflow.com/cc-pintel/car-counting) (MIT license)
- **Images**: 7,284 (416×416)
- **Classes**: Bus (0), Motorcycle (1), Pickup (2), SUV (3), Sedan (4), Truck (5), Van (6)
- **Splits**: train (5,536), valid (1,456), test (291)

## Requirements

```
ultralytics==8.4.106
opencv-python==5.0.0.93
torch==2.11.0
matplotlib==3.11.1
PyYAML==6.0.3
gradio==6.20.0
lap==0.5.13
```

See `requirements.txt` for the full list.
