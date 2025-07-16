"""
ishakdolek
15.08.2024
Ref: https://docs.ultralytics.com/tasks/segment/#models

"""

from ultralytics import YOLO


model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights


# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="yolo/coco8-seg.yaml", epochs=100, imgsz=640)
