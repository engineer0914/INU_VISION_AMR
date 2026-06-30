import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


MODEL_PATH = r"C:\Users\HDL\OneDrive\문서\LAB\2026 로보컵\비전모델\Model_s_ver2.0\best.pt"

CAMERA_ID = 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

IMG_SIZE = 640
CONF_THRES = 0.50
IOU_THRES = 0.50

USE_RETINA_MASKS = True
LINE_WIDTH = 2

SAVE_DIR = r"C:\Users\HDL\OneDrive\문서\LAB\2026 로보컵\비전모델\RealTimeResults"


def get_device():
    if torch.cuda.is_available():
        return 0
    return "cpu"


def open_camera(camera_id, width, height, fps):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다. CAMERA_ID={camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def draw_info_panel(frame, fps_value, device, conf, model_path, detected_count):
    text_lines = [
        f"FPS: {fps_value:.1f}",
        f"Device: {device}",
        f"Camera ID: {CAMERA_ID}",
        f"Conf Threshold: {conf:.2f}",
        f"Detected Objects: {detected_count}",
        "Q or ESC: Quit",
        "S: Save Screenshot",
        f"Model: {Path(model_path).name}"
    ]

    x0, y0 = 10, 10
    line_h = 24
    panel_w = 480
    panel_h = line_h * len(text_lines) + 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    for i, text in enumerate(text_lines):
        y = y0 + 25 + i * line_h
        cv2.putText(
            frame,
            text,
            (x0 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return frame


def print_model_info(device):
    print("=" * 70)
    print("YOLOv8 Real-time Segmentation")
    print("=" * 70)
    print(f"Model path      : {MODEL_PATH}")
    print(f"Camera ID       : {CAMERA_ID}")
    print(f"Device          : {device}")

    if torch.cuda.is_available():
        print(f"GPU name        : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU name        : CPU mode")

    print(f"Image size      : {IMG_SIZE}")
    print(f"Conf threshold  : {CONF_THRES}")
    print(f"IoU threshold   : {IOU_THRES}")
    print("=" * 70)
    print("Confidence 0.90 미만 객체는 표시하지 않습니다.")
    print("Q 또는 ESC: 종료")
    print("S: 현재 화면 저장")
    print("=" * 70)


def get_detected_count(result):
    if result.boxes is None:
        return 0

    if result.boxes.conf is None:
        return 0

    return int(len(result.boxes.conf))


def main():
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    use_half = True if device != "cpu" else False

    model = YOLO(str(model_path))

    print_model_info(device)

    cap = open_camera(
        camera_id=CAMERA_ID,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS
    )

    window_name = "YOLOv8 Duplo Segmentation - Webcam"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps_value = 0.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("프레임을 읽지 못했습니다.")
            break

        frame_idx += 1

        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            device=device,
            half=use_half,
            retina_masks=USE_RETINA_MASKS,
            verbose=False
        )

        result = results[0]
        detected_count = get_detected_count(result)

        annotated = result.plot(
            boxes=True,
            masks=True,
            labels=True,
            conf=True,
            line_width=LINE_WIDTH
        )

        now = time.time()
        dt = now - prev_time
        prev_time = now

        if dt > 0:
            current_fps = 1.0 / dt
            if fps_value == 0.0:
                fps_value = current_fps
            else:
                fps_value = 0.9 * fps_value + 0.1 * current_fps

        annotated = draw_info_panel(
            annotated,
            fps_value=fps_value,
            device=device,
            conf=CONF_THRES,
            model_path=MODEL_PATH,
            detected_count=detected_count
        )

        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q") or key == 27:
            break

        if key == ord("s") or key == ord("S"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = save_dir / f"webcam_result_{timestamp}_{frame_idx:06d}.png"
            cv2.imwrite(str(save_path), annotated)
            print(f"저장 완료: {save_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()