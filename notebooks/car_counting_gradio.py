import tempfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
COUNTING_OUTPUT_DIR = Path(tempfile.gettempdir()) / "car_counting"
COUNTING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_DIR = PROJECT_ROOT / "outputs" / "counting"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

LINE_Y_POSITION = 0.5
CAR_KEYWORDS = ["sedan", "suv", "pickup"]

import torch
import cv2
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")
DEVICE = 0 if torch.cuda.is_available() else "cpu"

#Load best.pt model

import gradio as gr

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}.\n"
        "Run car_training.ipynb first to train and save the model."
    )

model = YOLO(str(MODEL_PATH))
print(f"Model loaded from: {MODEL_PATH}")

car_class_ids = [
    i for i, name in model.names.items()
    if any(keyword in name.lower() for keyword in CAR_KEYWORDS)
]
if not car_class_ids:
    raise ValueError(
        f"No car classes found in model names: {model.names}.\n"
        f"Expected keywords: {CAR_KEYWORDS}"
    )
car_class_names = {i: model.names[i] for i in car_class_ids}
print(f"Car class IDs for counting: {car_class_ids}")
print(f"Car class names: {car_class_names}")

#Car Counting Function Logic

def count_cars(video_path, model, car_ids, line_y_pos=0.5, device="cpu", output_dir=None):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_dir is None:
        output_dir = COUNTING_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"car_counting_{video_path.stem}_{timestamp}.mp4"
    output_path = output_dir / output_name

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    LINE_Y = int(height * line_y_pos)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    prev_centers = {}
    counted_ids = set()
    total_count = 0
    frame_idx = 0

    print(f"Processing: {video_path.name}")
    print(f"  Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")
    print(f"  Counting line at Y={LINE_Y} ({line_y_pos * 100:.0f}% of height)")
    print(f"  Car class IDs: {car_ids}")

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            device=device, classes=car_ids, verbose=False
        )

        annotated = results[0].plot()

        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()

            for track_id, (x1, y1, x2, y2) in zip(track_ids, xyxy):
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)
                cv2.putText(annotated, f"ID:{track_id}", (cx + 6, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                prev_cy = prev_centers.get(track_id, cy)

                if prev_cy < LINE_Y and cy >= LINE_Y:
                    if track_id not in counted_ids:
                        total_count += 1
                        counted_ids.add(track_id)

                prev_centers[track_id] = cy

        cv2.line(annotated, (0, LINE_Y), (width, LINE_Y), (0, 0, 255), 2)
        cv2.putText(annotated, f"CARS COUNTED: {total_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(annotated)

        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            progress = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f"  Frame {frame_idx}/{total_frames} ({progress:.0f}%), "
                  f"Count: {total_count}, Elapsed: {elapsed:.1f}s")

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"  Total frames processed: {frame_idx}")
    print(f"  Total cars counted:     {total_count}")
    print(f"  Time elapsed:           {elapsed:.1f}s")
    print(f"  Output:                 {output_path}")

    return str(output_path), total_count

# Process video function
def process_video(video_file):
    if video_file is None:
        return None, "No video uploaded."

    try:
        output_path, car_count = count_cars(
            video_path=video_file,
            model=model,
            car_ids=car_class_ids,
            line_y_pos=LINE_Y_POSITION,
            device=DEVICE,
            output_dir=COUNTING_OUTPUT_DIR,
        )
        shutil.copy2(output_path, SAVE_DIR / Path(output_path).name)
        count_text = f"Total cars counted: {car_count}"
        return output_path, count_text
    except Exception as e:
        return None, f"Error processing video: {str(e)}"

#Gradio Interface

with gr.Blocks(title="Car Counting System") as demo:
    gr.Markdown("# Car Counting System")
    gr.Markdown(
        "Upload a traffic video to detect, track, and count cars "
        "using **YOLO11s** and **ByteTrack**."
    )

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Upload Traffic Video")
            process_btn = gr.Button("Process Video", variant="primary")

        with gr.Column():
            output_video = gr.Video(label="Processed Video", autoplay=False)
            count_display = gr.Textbox(label="Car Count", value="Waiting for video...")

    process_btn.click(
        fn=process_video,
        inputs=[input_video],
        outputs=[output_video, count_display],
    )

    gr.Markdown(
        "### How it works\n"
        "1. Upload a traffic video\n"
        "2. Click Process Video\n"
        "3. YOLO11s detects cars\n"
        "4. ByteTrack assigns tracking IDs\n"
        "5. A virtual line detects crossings\n"
        "6. Each car is counted once\n"
        "7. Processed video and count are displayed\n"
        "\n"
        "© MALVO"
    )

# Launch Gradio

if __name__ == "__main__":
    demo.launch()
