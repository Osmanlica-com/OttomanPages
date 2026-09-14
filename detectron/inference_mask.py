import os
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

# Set up config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

# Use your trained model weights
cfg.MODEL.WEIGHTS = "/home/aidigital/batuhan/detectron_output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this inference
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # number of classes you trained for
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Register metadata if needed
MetadataCatalog.get("textblock_val").set(thing_classes=["TextBlock"])

# Create predictor
predictor = DefaultPredictor(cfg)

def predict(image):
    # Load and run inference on an image
    outputs = predictor(image)
    return outputs

# # Visualize results
# v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("textblock_val"), scale=1.0)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# result_image = out.get_image()[:, :, ::-1]

# # Save or display result
# cv2.imwrite("output_prediction.jpg", result_image)
# OR: cv2.imshow("Prediction", result_image); cv2.waitKey(0)
