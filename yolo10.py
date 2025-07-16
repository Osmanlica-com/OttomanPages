from ultralytics import YOLO

# Load a pre-trained YOLOv10n model
model = YOLO("yolov10n.pt")

# Train the model
model.train(resume=True, data="yolo/coco8-seg.yaml",
            epochs=100, imgsz=640, workers=0)
