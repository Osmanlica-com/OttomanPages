"""
Bismillah
ishakdolek
15.08.2024
Ref: https://docs.ultralytics.com/tasks/segment/#models
"""

from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO(model="yolo11x-seg.pt")
# #model = YOLO("yolo11n-seg.pt")
# # Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="config.yaml", epochs=100, imgsz=640, save_period=5)