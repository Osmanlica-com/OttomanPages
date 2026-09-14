# OttomanPages

**OttomanPages** is a research framework and benchmark for **page layout analysis and segmentation of historical Ottoman Turkish documents**. The project focuses on automatically identifying textual regions in Ottoman manuscripts, archival documents, newspapers, and other historical document images.

Historical Ottoman documents present substantial challenges for document image analysis because of their heterogeneous layouts, multi-column structures, handwritten and printed text, decorative elements, skewed text regions, physical degradation, and complex page compositions. These characteristics make conventional document segmentation methods developed primarily for modern Latin-script documents less effective.

OttomanPages provides a dedicated dataset, experimental implementations, and evaluation results for investigating these challenges.

---

## Overview

Page segmentation is an important first step in historical document analysis. Accurate identification of text regions can substantially improve subsequent tasks such as:

* Optical Character Recognition (OCR)
* Handwritten Text Recognition (HTR)
* Line segmentation
* Document layout analysis
* Text extraction
* Historical document transcription
* Digital archival processing

The **OttomanPages** project investigates page-level text-region segmentation specifically for Ottoman Turkish documents.

The repository contains implementations and experiments using several state-of-the-art segmentation approaches, including:

* **YOLO-based segmentation**
* **DeepLabV3**
* **Detectron2**
* **PaddleDetection**
* **Segment Anything Model (SAM)**
* **SegFormer**

The experiments are designed to provide a common benchmark for comparing these approaches on Ottoman document layouts.

---

## Dataset

The OttomanPages dataset consists of historical Ottoman document pages annotated for page layout segmentation.

The dataset contains documents representing different Ottoman document types and page layouts, including:

* Newspapers
* Kadi registry books
* Population registers
* Manuscripts and other historical documents

The annotations identify textual regions on document pages and can be used for training and evaluating page segmentation models.

### Dataset Access

The dataset is publicly available through Zenodo:

**DOI:** https://doi.org/10.5281/zenodo.22756788

The dataset should be cited separately when it is used in research.

---

## Repository Structure

```text
OttomanPages/
│
├── analysis/
│   └── Dataset analysis and evaluation scripts
│
├── deeplabv3/
│   └── DeepLabV3 experiments
│
├── detectron/
│   └── Detectron2 experiments
│
├── paddledetection/
│   └── PaddleDetection experiments
│
├── result/
│   └── Experimental results and outputs
│
├── sam/
│   └── Segment Anything Model experiments
│
├── segformer/
│   └── SegFormer experiments
│
├── yolo/
│   └── YOLO-based segmentation experiments
│
├── req.txt
│   └── Python dependencies
│
└── README.md
```

---

## Models

OttomanPages provides a comparative experimental environment for several segmentation architectures.

### YOLO

YOLO-based segmentation models are evaluated for their ability to detect and segment text regions in Ottoman pages.

The YOLO experiments investigate different model configurations and provide segmentation metrics for comparison with other approaches.

### DeepLabV3

DeepLabV3 is included as a semantic segmentation baseline. It provides a classical deep-learning baseline for evaluating pixel-level page segmentation.

### Detectron2

Detectron2-based models are included to evaluate region-based object detection and segmentation approaches on Ottoman document layouts.

### PaddleDetection

PaddleDetection provides an additional object detection and segmentation framework for comparison with the other approaches.

### Segment Anything Model

The Segment Anything Model (SAM) is evaluated to investigate the applicability of foundation segmentation models to historical Ottoman documents.

### SegFormer

SegFormer is included as a transformer-based semantic segmentation architecture and provides a modern transformer baseline for the task.

---

## Experimental Pipeline

The general experimental workflow is:

```text
Ottoman Document Images
          │
          ▼
   Dataset Preparation
          │
          ▼
   Page-Level Annotation
          │
          ▼
   Train / Test Split
          │
          ▼
 ┌────────┴─────────┐
 │                  │
 ▼                  ▼
YOLO            DeepLabV3
 │                  │
 ├── Detectron2     │
 ├── PaddleDetection│
 ├── SAM            │
 └── SegFormer      │
          │
          ▼
   Segmentation Results
          │
          ▼
     Evaluation
```

The resulting segmentation masks and predictions can subsequently be used as input to downstream OCR and HTR pipelines.

---

## Evaluation

The models are evaluated using standard segmentation and detection metrics, including:

* Precision
* Recall
* F1-score
* mAP@50
* mAP@50–95
* Intersection over Union (IoU)

These metrics allow both detection quality and segmentation quality to be compared across different approaches.

For historical documents, quantitative evaluation is particularly important because visually similar layouts can have substantially different segmentation difficulty.

---

## Page Layout Complexity

Ottoman document pages can vary considerably in structural complexity. A page containing a single regular text block is substantially easier to segment than a page containing multiple columns, irregular text regions, touching regions, skewed lines, or highly variable text heights.

OttomanPages therefore supports further analysis of segmentation performance according to page-layout characteristics.

Potential complexity factors include:

* Number of text regions
* Text-region density
* Region touching/overlap
* Text-height variation
* Text-line angle variation
* Baseline/line-spacing variation
* Multi-column structure
* Irregular page layouts

This enables model performance to be analyzed not only globally, but also according to the structural complexity of individual pages.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Osmanlica-com/OttomanPages.git
cd OttomanPages
```

Create a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r req.txt
```

> **Note:** Some experiments depend on external frameworks such as Detectron2 and DHSegment. GPU-enabled installations may require additional system- and CUDA-specific configuration.

---

## Requirements

The repository uses a Python-based deep-learning environment with packages for:

* PyTorch
* Transformers
* Hugging Face Datasets
* Albumentations
* OpenCV
* Ultralytics
* Detectron2
* PaddleDetection
* SegFormer
* SAM
* DeepLabV3
* Evaluation and visualization utilities

The complete dependency specification is provided in [`req.txt`](req.txt).

---

## Reproducibility

To reproduce an experiment:

1. Download the OttomanPages dataset.
2. Prepare the dataset according to the directory structure expected by the corresponding model.
3. Install the dependencies listed in `req.txt`.
4. Select the desired segmentation framework.
5. Train or evaluate the model.
6. Store predictions and evaluation results in the corresponding experiment directory.
7. Use the analysis scripts to compare model performance.

Each model implementation is maintained in its own directory to make the experiments easier to reproduce and extend.

---

## Research Applications

OttomanPages is intended to support research in:

* Digital Humanities
* Ottoman Studies
* Historical Document Analysis
* Document Image Analysis
* Optical Character Recognition
* Handwritten Text Recognition
* Historical OCR/HTR
* Computer Vision
* Layout Analysis
* Multilingual and non-Latin document processing

The project is particularly relevant to the development of digital tools for large-scale Ottoman archival collections.

---

## Citation

If you use the OttomanPages dataset, code, or experimental results in your research, please cite the corresponding publication and dataset.

### Dataset

```bibtex
@dataset{ottomanpages,
  title        = {OttomanPages},
  doi          = {10.5281/zenodo.22756788},
  publisher    = {Zenodo}
}
```

### Project Repository

```bibtex
@misc{ottomanpages,
  title        = {OttomanPages: Page Segmentation for Historical Ottoman Documents},
  author       = {Osmanlica.com},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/Osmanlica-com/OttomanPages}
}
```

> **Note:** Replace the citation metadata above with the final bibliographic information of the associated paper once the publication details are available.

---

## Related Work

OttomanPages is part of a broader effort to develop computational resources for Ottoman Turkish document processing, including page segmentation, line segmentation, OCR/HTR, transliteration, and text analysis.

The project is developed within the **Osmanlica.com** ecosystem.

---

## Contributing

Contributions, suggestions, and discussions are welcome.

If you find a problem with the dataset, implementation, or documentation, please open an issue in this repository.

For code contributions:

```bash
git checkout -b feature/my-new-feature
git commit -am "Add new feature"
git push origin feature/my-new-feature
```

Then open a pull request.

---

## License

Please refer to the repository and dataset license information before redistributing the code or dataset.

The dataset hosted on Zenodo may have licensing conditions that are separate from the source code in this repository.

---

## Acknowledgements

We thank the researchers, Ottoman Turkish experts, and institutions contributing to the digitization, annotation, and computational analysis of Ottoman historical documents.

This work contributes to the development of openly accessible computational resources for Ottoman Turkish and historical document analysis.

---

## Contact

For questions, collaboration, or research inquiries, please use the GitHub repository:

**https://github.com/Osmanlica-com/OttomanPages**

Project website:

**https://osmanlica.com**
