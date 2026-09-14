import json
import os

dataset_path = "dataset"
split = "train"

split_path = os.path.join(dataset_path, split)

with open(os.path.join(split_path, f"layout_{split}_dataset.json"), "r", encoding="utf8") as ds_file:
    dataset_old = json.loads(ds_file.read())

annotations = dataset_old["annotations"]

dataset = {}

for annotation in annotations:
    id = annotation["image_id"]
    path = f"{id}.png"
    try:
        file_size = os.stat(os.path.join(split_path, path)).st_size
    except Exception as e:
        print(e)
        continue
    points = annotation["bbox"]
    region_object = {
        "shape_attributes": {
            "name": "polygon",
            "all_points_x": [point[0] for point in points],
            "all_points_y": [point[1] for point in points],
        },
        "region_attributes": {},
    }
    if dataset.get(id):
        latest_region_key = max([int(key) for key in dataset[id]["regions"].keys()])
        dataset[id]["regions"][latest_region_key + 1] = region_object
    else:
        dataset[id] = {
            "fileref": "",
            "size": file_size,
            "filename": path,
            "base64_img_data": "",
            "file_attributes": {},
            "regions": {0: region_object},
        }

with open(f"{split}.json", 'w', encoding="utf8") as newfile:
    newfile.write(json.dumps(dataset, ensure_ascii=False))
# print(dataset)