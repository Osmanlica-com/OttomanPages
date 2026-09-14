# Ottoman page-layout baseline evaluation

- Ground truth: `data/test/images/layout_test_dataset.json` (509 pages, 12761 TextBlock regions)
- Class-agnostic: predicted class labels are ignored, every prediction is a candidate region. The COCO-pretrained baselines have no TextBlock class.
- IoU is computed on axis-aligned boxes; ground-truth polygons are reduced to their bounding box (identical to the dataset's own `.txt` export).
- Precision / Recall / F1 / Mean IoU at IoU>=0.5 and confidence>=0.5. Mean IoU averages matched true positives.
- mAP via COCOeval, single class, maxDets=1000.

## Experiment 1 - base models, full test set

Pre-trained weights, no fine-tuning.

| Model | Precision@50 | Recall@50 | F1@50 | mAP@50 | mAP@50-95 | Mean IoU |
|---|---|---|---|---|---|---|
| DeepLabV3 | 0.0745 | 0.0059 | 0.0109 | 0.0018 | 0.0008 | 0.6543 |
| SAM (large) | 0.1858 | 0.2124 | 0.1982 | 0.0662 | 0.0289 | 0.7281 |
| SAM (base) | 0.1073 | 0.0885 | 0.0970 | 0.0169 | 0.0070 | 0.6852 |
| YOLO8n-seg | 0.0408 | 0.0002 | 0.0003 | 0.0049 | 0.0019 | 0.9465 |
| Yolov10n-det | 0.0357 | 0.0005 | 0.0011 | 0.0114 | 0.0065 | 0.8247 |
| YOLO11x-seg | 0.0547 | 0.0011 | 0.0022 | 0.0084 | 0.0054 | 0.7187 |
| Detectron2 | 0.0081 | 0.0002 | 0.0003 | 0.0004 | 0.0002 | 0.8513 |
| PaddleDetection | 0.0979 | 0.0059 | 0.0111 | 0.0076 | 0.0023 | 0.6613 |

## Experiment 2 - fine-tuned models, by document type

Weights from `weights/trained/`.

### Kadi Registry Book (93 pages)

| Model | Precision@50 | Recall@50 | F1@50 | mAP@50 | mAP@50-95 | Mean IoU |
|---|---|---|---|---|---|---|
| YOLO8n-seg | 0.9706 | 0.9481 | 0.9592 | 0.9799 | 0.8426 | 0.9145 |
| Yolov10n-det | 0.9537 | 0.9577 | 0.9557 | 0.9773 | 0.8311 | 0.9080 |
| YOLO11x-seg | 0.9665 | 0.9454 | 0.9558 | 0.9761 | 0.8353 | 0.9158 |
| Detectron2 | 0.9214 | 0.9604 | 0.9405 | 0.9722 | 0.8026 | 0.9027 |
| PaddleDetection | 0.9130 | 0.6311 | 0.7464 | 0.8278 | 0.5627 | 0.8585 |

### Newspaper (368 pages)

| Model | Precision@50 | Recall@50 | F1@50 | mAP@50 | mAP@50-95 | Mean IoU |
|---|---|---|---|---|---|---|
| YOLO8n-seg | 0.9203 | 0.8720 | 0.8955 | 0.9368 | 0.7164 | 0.8753 |
| Yolov10n-det | 0.9146 | 0.8762 | 0.8950 | 0.9316 | 0.7048 | 0.8744 |
| YOLO11x-seg | 0.9218 | 0.8741 | 0.8973 | 0.9341 | 0.7182 | 0.8792 |
| Detectron2 | 0.8766 | 0.9321 | 0.9035 | 0.9423 | 0.6802 | 0.8569 |
| PaddleDetection | 0.7577 | 0.4877 | 0.5934 | 0.5802 | 0.3377 | 0.8091 |

### Population Registry Book (48 pages)

| Model | Precision@50 | Recall@50 | F1@50 | mAP@50 | mAP@50-95 | Mean IoU |
|---|---|---|---|---|---|---|
| YOLO8n-seg | 0.9647 | 0.9399 | 0.9521 | 0.9616 | 0.4956 | 0.7648 |
| Yolov10n-det | 0.9671 | 0.9400 | 0.9533 | 0.9587 | 0.4913 | 0.7641 |
| YOLO11x-seg | 0.9645 | 0.9400 | 0.9521 | 0.9605 | 0.4947 | 0.7655 |
| Detectron2 | 0.9548 | 0.8867 | 0.9195 | 0.8942 | 0.3912 | 0.7361 |
| PaddleDetection | 0.5950 | 0.1441 | 0.2321 | 0.1622 | 0.0598 | 0.7050 |

### Overall

| Model | Precision@50 | Recall@50 | F1@50 | mAP@50 | mAP@50-95 | Mean IoU |
|---|---|---|---|---|---|---|
| YOLO8n-seg | 0.9503 | 0.9174 | 0.9336 | 0.9501 | 0.5981 | 0.8092 |
| Yolov10n-det | 0.9487 | 0.9194 | 0.9339 | 0.9456 | 0.5904 | 0.8083 |
| YOLO11x-seg | 0.9504 | 0.9180 | 0.9339 | 0.9491 | 0.5980 | 0.8110 |
| Detectron2 | 0.9241 | 0.9063 | 0.9151 | 0.9094 | 0.5248 | 0.7882 |
| PaddleDetection | 0.7139 | 0.2882 | 0.4107 | 0.3373 | 0.1785 | 0.7838 |
