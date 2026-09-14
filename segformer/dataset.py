import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from constants import label2id, id2label
import json

class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor

        annotation_file_path = os.path.join(self.root_dir, "via_region_data.json")

        with open(annotation_file_path, "r", encoding="utf8") as annotation_file:
            self.annotation_object = json.loads(annotation_file.read())

        self.annotations = []

        for img_id, annotation in self.annotation_object.items():
            annotations = []
            for region in annotation["regions"].values():
                merged_points = [list(point) for point in zip(region["shape_attributes"]["all_points_x"], region["shape_attributes"]["all_points_y"])]
                annotations.append(merged_points)
            annotation_object = {
                "img_path": annotation["filename"],
                "annotations":annotations
            }
            self.annotations.append(annotation_object)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations[idx]
        annotations = data["annotations"]

        image = Image.open(os.path.join(self.root_dir, data["img_path"])).convert("RGB")
        img_array = np.array(image)
        annotation_2d = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8) # height, width
        for annotation in annotations:
                    # Extract coordinates
            x_coords = [point[0] for point in annotation]
            y_coords = [point[1] for point in annotation]
            
            # Calculate the rectangular region for the bounding box
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            annotation_2d[y_min:y_max, x_min:x_max] = label2id["ottoman_text"]
        
        # mask_image = Image.fromarray(annotation_2d)
        # mask_image.save("annotation.png")

        # randomly crop + pad both image and segmentation map to same size
        # feature extractor will also reduce labels!
        encoded_inputs = self.feature_extractor(image, Image.fromarray(annotation_2d), return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs