"""
Assignment 9: YOLOv8 Training / Fine-Tuning Script
---------------------------------------------------
Fine-tunes a pretrained YOLOv8n model on the COCO128 dataset (auto-downloads)
and prints Precision, Recall, and mAP metrics.

To use Open Images V7 instead, change the 'data' parameter to 'open-images-v7.yaml'.
Note: Open Images is very large (~561 classes, millions of images). For a quicker
demo, COCO128 (128 images, 80 classes) is used by default.
"""

from ultralytics import YOLO


def main():
    # ------------------------------------------------------------------
    # 1. Load a pretrained YOLOv8 model (auto-downloads weights ~6 MB)
    # ------------------------------------------------------------------
    model = YOLO("yolov8n.pt")  # 'n' = nano (smallest & fastest)
    print("Pretrained YOLOv8n model loaded successfully.\n")

    # ------------------------------------------------------------------
    # 2. Fine-tune on COCO128 (auto-downloads dataset on first run)
    #    Change data='open-images-v7.yaml' for Open Images V7.
    # ------------------------------------------------------------------
    results = model.train(
        data="coco128.yaml",   # dataset config (auto-downloads)
        epochs=25,             # number of training epochs
        imgsz=640,             # input image size
        batch=16,              # batch size (lower if GPU memory is limited)
        name="yolov8_finetuned",
        patience=5,            # early stopping patience
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 3. Evaluate on validation set — Precision, Recall, mAP
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION METRICS")
    print("=" * 60)

    metrics = model.val()

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    print(f"  Precision (mean) : {precision:.4f}")
    print(f"  Recall (mean)    : {recall:.4f}")
    print(f"  mAP@0.50         : {map50:.4f}")
    print(f"  mAP@0.50:0.95    : {map50_95:.4f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 4. Export the fine-tuned model path for use in inference / app
    # ------------------------------------------------------------------
    best_model_path = results.save_dir / "weights" / "best.pt"
    print(f"\nBest model saved at: {best_model_path}")
    print("Use this path in inference.py and app.py for best results.")


if __name__ == "__main__":
    main()