import numpy as np
import os
import json
from PIL import Image
import cv2
import shutil

dataset_path = "/workspace/dataset"
new_dataset_path = "/workspace/masked_dataset"

os.makedirs(new_dataset_path, exist_ok=True)

def create_mask(image_path, bboxes):
    # Load the image to get its dimensions
    image = Image.open(image_path)
    width, height = image.size
    
    # Create an empty mask (black background)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Convert bounding boxes into numpy array
    for bbox in bboxes:
        polygon = np.array(bbox, np.int32)
        polygon = polygon.reshape((-1, 1, 2))  # Reshape for OpenCV
        
        # Fill the polygon with white (value = 255)
        cv2.fillPoly(mask, [polygon], 255)
    return mask

for segment in ["train", "test"]:
    segment_path = os.path.join(dataset_path, segment)
    new_segment_path = os.path.join(new_dataset_path, segment)
    new_segment_masked_path = os.path.join(new_dataset_path, segment + "_masked")
    os.makedirs(new_segment_path, exist_ok=True)
    os.makedirs(new_segment_masked_path, exist_ok=True)
    data_file = os.path.join(segment_path, f"layout_{segment}_dataset.json")
    with open(data_file, "r", encoding="utf8") as datafile:
        data = json.load(datafile)
    for image in data["images"]:
        image_id = image["id"]
        bboxes = []
        for annotation in [
            ann for ann in data["annotations"] if ann["image_id"] == image_id
        ]:
            bboxes.append(annotation["bbox"])
        image_path = os.path.join(segment_path, f"{image_id}.png")
        if os.path.exists(image_path):
            mask_img = create_mask(image_path, bboxes)
            cv2.imwrite(os.path.join(new_segment_masked_path, f"{image_id}.png"), mask_img)
            shutil.copyfile(image_path, os.path.join(new_segment_path, f"{image_id}.png"))
