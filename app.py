"""Drone tracking demo.

Upload a video, and a trained YOLO model detects and tracks drones frame by frame.
Usage:
    python app.py
Set MODEL_PATH (env variable) to point to your trained weights.
"""

import os
import tempfile

import cv2
import gradio as gr
from ultralytics import YOLO

# ---------- CONFIG ----------
MODEL_PATH = os.getenv("MODEL_PATH", "runs/detect/drone_detect/weights/best.pt")
CONF_THRESHOLD = 0.4
IMGSZ = 1280
TRACKER = "botsort.yaml"  # "bytetrack.yaml" can also be used

model = YOLO(MODEL_PATH)


def _open_writer(out_path, fps, w, h):
    """Try to open a browser-playable (H.264) writer, fall back to mp4v."""
    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer
    raise RuntimeError("Could not open video writer (no codec found).")


def track_video(video_file, conf, imgsz, progress=gr.Progress()):
    if video_file is None:
        return None, "Please upload a video first."

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = temp_out.name
    temp_out.close()

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise gr.Error("Could not open the video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()

    out = _open_writer(out_path, fps, w, h)

    track_ids_seen = set()
    count = 0

    results_gen = model.track(
        source=video_file,
        conf=conf,
        imgsz=int(imgsz),
        tracker=TRACKER,
        stream=True,
        persist=True,
        verbose=False,
    )

    for r in results_gen:
        out.write(r.plot())

        if r.boxes is not None and r.boxes.id is not None:
            for tid in r.boxes.id.tolist():
                track_ids_seen.add(int(tid))

        count += 1
        progress(min(count / total, 1.0), desc=f"Processing {count}/{total}")

    out.release()

    summary = f"Processed {count} frames. Unique drones tracked: {len(track_ids_seen)}"
    return out_path, summary


# ---------- DARK THEME ----------
custom_css = """
.gradio-container { background-color: #0f1115 !important; font-family: 'Inter', 'Segoe UI', sans-serif !important; }
.custom-header { text-align: center; padding: 20px 0 8px 0; }
.custom-header h1 { font-weight: 600; color: #f1f5f9; margin: 0; font-size: 26px; }
.custom-header hr { border: 0; height: 2px; background: linear-gradient(to right, transparent, #22c55e, transparent); width: 30%; margin: 10px auto; }
.gr-input, .gr-output, .gr-box, .block { background: #1a1d24 !important; border: 1px solid #2a2e37 !important; border-radius: 12px !important; }
.gr-button-primary { background: #22c55e !important; border: none !important; border-radius: 10px !important; padding: 10px 20px !important; font-weight: 700 !important; color: #0f1115 !important; }
.gr-button-primary:hover { background: #16a34a !important; }
label { color: #e2e8f0 !important; font-weight: 500 !important; }
body, .gradio-container, .gr-markdown, span, div { color: #e2e8f0 !important; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
    gr.HTML("""
        <div class="custom-header">
            <h1>Drone Tracking</h1>
            <hr>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_vid = gr.Video(label="Input Video", sources=["upload"], height=420)
            conf_slider = gr.Slider(0.1, 0.9, value=CONF_THRESHOLD, step=0.05, label="Confidence threshold")
            imgsz_slider = gr.Slider(640, 1280, value=IMGSZ, step=64, label="Image size (imgsz)")
            btn = gr.Button("Track Video", variant="primary")
            output_text = gr.Textbox(label="Summary", lines=2)
        with gr.Column(scale=1):
            output_vid = gr.Video(label="Tracking Result", height=420)

    btn.click(
        fn=track_video,
        inputs=[input_vid, conf_slider, imgsz_slider],
        outputs=[output_vid, output_text],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=False)
