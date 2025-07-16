"""
ishakdolek
15.08.2024
Ref: https://docs.ultralytics.com/tasks/segment/#models

"""

from ultralytics import YOLO

# build from YAML and transfer weights
model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt")

# Train the model
results = model.train(data="yolo/coco8-seg.yaml", epochs=100, cache=False)
