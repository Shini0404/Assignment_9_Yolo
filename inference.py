"""
Assignment 9: YOLOv8 Inference Script
--------------------------------------
Runs object detection on images or a video file and displays
bounding boxes, class labels, and confidence scores.

Usage:
    python inference.py                          # webcam
    python inference.py --source image.jpg       # single image
    python inference.py --source video.mp4       # video file
    python inference.py --source path/to/folder  # folder of images
"""

import argparse
import cv2
from ultralytics import YOLO


def run_inference(source, model_path, confidence):
    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")

    # Run prediction
    results = model.predict(
        source=source,
        conf=confidence,
        show=False,
        stream=True,
    )

    for result in results:
        annotated_frame = result.plot()

        # Print detections to console
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  [{cls_name}] conf={conf:.2f}  bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

        cv2.imshow("YOLOv8 Detection", annotated_frame)

        # Press 'q' to quit, any other key for next image
        key = cv2.waitKey(1 if source == "0" else 0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    print("Inference complete.")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Inference")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Image/video path, folder, or '0' for webcam (default: webcam)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLOv8 model weights (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum confidence threshold (default: 0.25)",
    )
    args = parser.parse_args()

    source = int(args.source) if args.source == "0" else args.source
    run_inference(source, args.model, args.confidence)


if __name__ == "__main__":
    main()