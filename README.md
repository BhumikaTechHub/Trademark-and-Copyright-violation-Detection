# AI Logo Verification Project  (on prototype experimentation stage)

A multi-technique pipeline for detecting fake/counterfeit brand logos and plagiarized text content using computer vision, deep learning, and NLP.

---

## Overview

This project addresses two categories of intellectual property violations:

1. **Trademark Violations** — Identifying counterfeit or imitation brand logos using image similarity techniques.
2. **Copyright Violations** — Detecting plagiarized textual content using NLP-based similarity models.

---

## Features

### Image-Based Trademark Detection
- **SSIM (Structural Similarity Index)** — Measures pixel-level structural similarity between genuine and fake logos.
- **ORB Feature Matching** — Detects and matches keypoints between two logos using binary descriptors.
- **Perceptual Hashing (dHash & pHash)** — Compares logos robustly even when they've been resized, filtered, or lightly edited.
- **TF-IDF + Cosine Similarity** — Compares brand name text strings for textual trademark similarity.
- **YOLOv8 Object Detection** — Trains a custom logo detector and runs inference on brand image datasets.
- **ResNet50 Siamese-style Comparison** — Extracts deep feature embeddings and computes Euclidean distance for robust similarity scoring.

### Text-Based Copyright Detection
- **Sentence-BERT (`all-MiniLM-L6-v2`)** — Encodes source and suspicious documents into dense embeddings.
- **Logistic Regression & Random Forest** — Binary classifiers trained on cosine similarity scores to label plagiarism.
- **Evaluation Metrics** — Classification report, accuracy, confusion matrix, and ROC AUC score.

### Supporting Utilities
- Web scraping for brand image collection (`requests` + `BeautifulSoup`)
- Automated image annotation in YOLO format
- Image preprocessing pipeline (resize, grayscale conversion)
- Google Sheets integration for logging detection results
- Scheduled/periodic scanning using the `schedule` library
- Openverse API integration for sourcing public domain creative works

---

## Dataset

The notebook expects a zipped dataset at `/mnt/Dataset.zip` that extracts into two subdirectories:

```
extracted_dataset/
├── output/          # Fake/generated logos, organized by brand
└── genLogoOutput/   # Genuine logos, organized by brand
```

Sample brands used: **Apple**, **Burger King**, **Nike**

For text plagiarism, the [PlagBench](https://github.com/Brit7777/plagbench) dataset is cloned directly from GitHub.

---

## Requirements

Install dependencies before running:

```bash
pip install ultralytics opencv-python scikit-image imagehash Pillow matplotlib
pip install scikit-learn sentence-transformers seaborn pandas numpy scipy
pip install requests beautifulsoup4 schedule gspread tensorflow keras
```

Or install individually as prompted by each notebook section.

---

## Usage

The notebook is structured as sequential sections. Run cells in order.

### 1. Dataset Setup
Unzip the dataset and verify brand category listings (genuine vs. fake).

### 2. SSIM & ORB Comparison
Load a genuine and fake logo pair, compute the SSIM score, and run ORB feature matching to visualize keypoint correspondences.

```
SSIM Score interpretation:
  ~1.0  → Identical images
  0.9+  → Very similar
  0.5–0.7 → Moderate similarity
  <0.5  → Visually different
```

### 3. Perceptual Hashing
Compute dHash and pHash similarity scores. Scores in the 0.6–0.8 range suggest possible edited or rebranded logos.

### 4. TF-IDF Text Similarity
Compare brand name strings using cosine similarity on TF-IDF vectors.

### 5. YOLOv8 Logo Detection
- Upload train and validation images to the appropriate folders.
- Auto-generate YOLO-format annotations.
- Train YOLOv8 on the custom dataset (`dataset.yaml`).
- Run inference and export detection results to CSV.

### 6. ResNet50 Feature Similarity
Load a ResNet50 backbone (pretrained on ImageNet), extract feature vectors, and compare genuine vs. fake logos using Euclidean distance. Lower distance = more similar.

### 7. Plagiarism Detection (Text)
- Load the PlagBench dataset.
- Encode document pairs with Sentence-BERT.
- Train Logistic Regression and Random Forest classifiers.
- Evaluate with accuracy, classification report, confusion matrix, and ROC AUC.

---

## Results Summary

| Technique | Use Case | Strength |
|---|---|---|
| SSIM | Logo structure comparison | Fast, interpretable |
| ORB | Keypoint matching | Handles rotation/scale |
| pHash / dHash | Resized/edited logos | Robust to minor edits |
| TF-IDF Cosine | Brand name text | Simple text matching |
| YOLOv8 | Logo detection in images | Localizes logos in context |
| ResNet50 Euclidean | Deep feature similarity | Handles distortion & color changes |
| SBERT + LR/RF | Text plagiarism | Semantic understanding |

---

## Conclusion

This multi-model pipeline improves robustness across a wide range of real-world violation scenarios — from near-identical logo copies to heavily distorted counterfeits and semantically similar plagiarized text.

---

## Future Work

- Integrate **CLIP embeddings** to jointly compare visual and textual brand features.
- Add **OCR** to extract and compare brand names embedded inside images.
- Extend to **audio watermark** detection.
- Implement **active learning** to retrain models on user-flagged violations.
- Deploy as a **REST API** for real-time trademark scanning.
