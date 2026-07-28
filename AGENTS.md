# AGENTS.md

## Project

YOLO-based car counting from video using YOLO11s + ByteTrack + Gradio.

## Key gotchas

- **Both notebooks must be run with `notebooks/` as the working directory.** They resolve `PROJECT_ROOT` via `Path.cwd().parent`. Running from any other directory will break all path resolution.
- **`requirements.txt` lists direct deps**, but `torch` installs CPU-only from PyPI. For GPU, install torch separately: `pip install torch --index-url https://download.pytorch.org/whl/cu128`
- **Training outputs go to `notebooks/runs/detect/car_training/`** (Ultralytics default, relative to notebook CWD). The notebook's final cell copies `best.pt` from there to `models/best.pt`.
- **Gradio `demo.launch()` will fail with `InvalidPathError`** because output videos land in `outputs/counting/` but Gradio's CWD is `notebooks/`. The notebook needs `demo.launch(allowed_paths=["../outputs/counting"])`. The `.py` script avoids this by writing to a temp directory instead.
- **`data.yaml` paths are relative to the yaml file's own directory**, not CWD. The notebook passes the absolute path to `data.yaml`, so this is handled correctly as long as the dataset directory hasn't been moved.

## Directories

| Dir                    | Purpose                                              |
| ---------------------- | ---------------------------------------------------- |
| `dataset/`             | Images + YOLO labels (train/valid/test), data.yaml    |
| `notebooks/`           | `car_training.ipynb`, `car_counting_gradio.ipynb`, `car_counting_gradio.py`    |
| `notebooks/runs/`      | Ultralytics training outputs (generated at runtime)   |
| `videos/`              | Input videos for inference                           |
| `models/`              | Saved model weights (`best.pt` after training)        |
| `outputs/`             | Inference outputs: `detection/`, `tracking/`, `counting/` |
| `.venv/`               | Python 3.14 virtual environment                       |

## Workflow

1. **`car_training.ipynb`** (run from `notebooks/`): fixes class labels (idempotent), verifies dataset, baseline tests pretrained YOLO11s, fine-tunes on custom data, copies `best.pt` to `models/best.pt`.
2. **`car_counting_gradio.ipynb`** (run from `notebooks/`): loads `models/best.pt`, tests ByteTrack tracking, implements line-crossing car counting, launches Gradio UI. **`car_counting_gradio.py`** is the standalone version — run from project root with `.venv\Scripts\python notebooks\car_counting_gradio.py`, open `http://127.0.0.1:7860`.

## Class labels

`dataset/data.yaml` has 7 clean classes:
`Bus, Motorcycle, Pickup, SUV, Sedan, Truck, Van`

The training notebook's class remap cell is idempotent — it detects mismatches and remaps original 10-class labels (with duplicates/typos) to 7, skips if already correct.

## Counting logic

- Counts passenger cars only: classes whose names contain `sedan`, `suv`, or `pickup` (case-insensitive) → Pickup (2), SUV (3), Sedan (4).
- ByteTrack via built-in `bytetrack.yaml` (no custom tracker config needed).
- One-directional line-crossing (top → bottom) at configurable fraction of frame height (default 0.5).
- Each track ID counted exactly once via `counted_ids` set.

## Environment

- `.venv` with Python 3.14. CUDA-capable GPU will be auto-detected and used if available; falls back to CPU otherwise.
- No CI, lint, tests, or build step exists.
