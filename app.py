"""
Assignment 9: Real-Time Object Detection Web App
-------------------------------------------------
FastAPI server that serves a webcam-based detection page.
The browser captures webcam frames via JavaScript, sends them
over WebSocket, and the server returns detection results drawn
on the frame.
Run:
uvicorn app:app --reload --host 0.0.0.0 --port 8000
Then open http://localhost:8000 in your browser.
"""
import base64
import json
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
app = FastAPI(title="YOLOv8 Real-Time Detection")
templates = Jinja2Templates(directory="templates")

# Load model once at startup (uses pretrained weights; change path
# to your fine-tuned model, e.g. "runs/detect/yolov8_finetuned/weights/best.pt")
model = YOLO("yolov8n.pt")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive base64-encoded JPEG frame from browser
            data = await websocket.receive_text()

            # Decode the image
            header, encoded = data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text(json.dumps({"detections": []}))
                continue

            # Run YOLOv8 inference
            results = model(frame, verbose=False)

            # Build detection list
            detections = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    detections.append({
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3]),
                        "confidence": round(float(box.conf[0]), 2),
                        "class": model.names[cls_id],
                    })

            # Also send back the annotated frame as base64
            annotated = results[0].plot()
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            annotated_b64 = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_text(json.dumps({
                "detections": detections,
                "frame": f"data:image/jpeg;base64,{annotated_b64}",
            }))
    except WebSocketDisconnect:
        print("Client disconnected")