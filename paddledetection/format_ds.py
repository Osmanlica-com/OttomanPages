import json
import os

dataset_dir = "/Users/batuhanmanav/Desktop/Ottoman-Train-Models/dataset"

categories = [
    {"supercategory": "", "id": 1, "name": "text"},
    {"supercategory": "", "id": 2, "name": "title"},
    {"supercategory": "", "id": 3, "name": "list"},
    {"supercategory": "", "id": 4, "name": "table"},
    {"supercategory": "", "id": 5, "name": "figure"},
]

def get_id_by_filename(filename):
    for image in images:
        if image["file_name"] == filename:
            return image["id"]

for part in ["train", "test"]:
    with open(os.path.join(dataset_dir, part, f"layout_{part}_dataset.json"), "r", encoding="utf8") as file:
        data = json.loads(file.read())

    images = []
    annotations = []

    ids = {}

    for id, image in enumerate(data["images"]):
    # file_name
    # width
    # id
    # height
        filename = f"{image['id']}.png"
        ids[filename] = id + 1
        images.append({
            "file_name":filename,
            "width":image['width'],
            "id":id + 1,
            "height":image['height'],
        })

    for aid, annotation in enumerate(data["annotations"]):
        polygon_points = annotation["bbox"]
        segmentation = []
        for point in polygon_points:
            segmentation.extend(point)  # Add x, y to the segmentation list
        segmentation.extend(polygon_points[0])  # Close the polygon by repeating the first point

        # Calculate bbox [x_min, y_min, width, height]
        x_coords = [point[0] for point in polygon_points]
        y_coords = [point[1] for point in polygon_points]
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        width = x_max - x_min
        height = y_max - y_min

        bbox = [x_min, y_min, width, height]
        annotations.append({
            "segmentation": [segmentation],
            "area": width * height,
            "iscrowd": 0,
            "bbox": bbox,
            "image_id": ids[f"{annotation["image_id"]}.png"],
            "category_id":1,
            "id":aid + 1
        })
    # segmentation [[]]
    # area
    # iscrowd : 0
    # image_id
    # bbox = [x1, y1, x2, y2]
    # category_id
    # id

    with open(f"{part}.json", "+w", encoding="utf8") as formatted_ds:
        formatted_ds.write(json.dumps({
            "images":images,
            "annotations":annotations,
            "categories":categories
        }))