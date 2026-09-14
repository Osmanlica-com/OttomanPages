import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import os
import cv2
import json
import numpy as np
from detectron2.structures import BoxMode

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog

# 👇 Define dataset parser for VIA format
def get_dataset_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for anno in annos.values():
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# 👇 Register datasets
for d in ["train", "val"]:
    DatasetCatalog.register("my_dataset_" + d, lambda d=d: get_dataset_dicts(f"dataset/{d}"))
    MetadataCatalog.get("my_dataset_" + d).set(thing_classes=["object"])

# 👇 Setup config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 2000
cfg.SOLVER.STEPS = []  # No LR decay
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only one class ("object")

# 👇 Output directory
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# 👇 Train
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
import os
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

dataset_dir = "/home/aidigital/batuhan/layout_dataset"

# Register the datasets (COCO format)
register_coco_instances("textblock_train", {}, os.path.join(dataset_dir, "all", "via_region_data.json"), os.path.join(dataset_dir, "all"))

# Load the config
cfg = get_cfg()

# Load base Mask R-CNN config
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

# Set custom dataset
cfg.DATASETS.TRAIN = ("textblock_train",)
cfg.DATALOADER.NUM_WORKERS = 2

# Use pretrained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000     # Adjust based on your dataset size
cfg.SOLVER.STEPS = []           # No LR decay

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only "TextBlock"

# Output directory
cfg.OUTPUT_DIR = "/home/aidigital/batuhan/detectron_output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Train
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
