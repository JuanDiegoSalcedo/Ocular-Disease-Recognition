# Ocular Disease Recognition

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)
![License](https://img.shields.io/badge/License-GPL--2.0-green)

Deep learning pipeline for **8-class retinal fundus image classification** using the ODIR-5K dataset.
The system classifies fundus photographs into one of eight diagnostic categories, from healthy retinas
to conditions including Diabetic Retinopathy, Glaucoma, and Age-related Macular Degeneration.

---

## Problem Statement

Retinal fundus photography provides a non-invasive window into ocular and systemic health. Automated
classification of fundus images can accelerate diagnosis in underserved clinical settings. This project
explores the full deep learning pipeline — from raw tabular metadata to a fine-tuned transfer learning
model — on a real clinical dataset with significant class imbalance.

The central challenge is that the dataset is **severely imbalanced**: Normal retinas account for 45%
of samples while Hypertension represents only 2%, making standard accuracy an unreliable performance
metric and requiring careful design of the training strategy.

---

## Dataset

**ODIR-5K** — Ocular Disease Intelligent Recognition  
Source: [Kaggle — andrewmvd/ocular-disease-recognition-odir5k](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

| Class | Code | Train | Test |
|---|---|---|---|
| Normal | N | 2298 | 575 |
| Diabetic Retinopathy | D | 1286 | 322 |
| Other diseases | O | 566 | 142 |
| Cataract | C | 234 | 59 |
| Glaucoma | G | 227 | 57 |
| Age-related Macular Degeneration | A | 213 | 53 |
| Pathological Myopia | M | 186 | 46 |
| Hypertension | H | 103 | 25 |
| **Total** | | **5113** | **1279** |

Images: 512×512 RGB JPEG fundus photographs (both eyes per patient). The dataset uses a stratified
80/20 train/test split (`random_state=511`).

---

## Methodology

```
full_df.csv
    │
    ├── 01_eda_and_preprocessing.ipynb
    │     ├─ Drop redundant columns (per-eye keywords, Kaggle filepath)
    │     ├─ Merge 8 binary flags → single diagnostic string
    │     ├─ Encode target as integer class index (0–7)
    │     ├─ Exploratory analysis: class distribution, age, gender, sample images
    │     └─ Stratified 80/20 split → train_data.h5 / test_data.h5
    │
    └── 02_model_training.ipynb
          ├─ Load images at 224×224 (float32 [0–255], EfficientNetB0 normalises internally)
          ├─ Build EfficientNetB0 + classification head
          ├─ Extract validation set (20% per class) from ORIGINAL images BEFORE augmentation
          ├─ Tiered augmentation: 0–9 copies per class (10 transforms) → 2.2:1 imbalance ratio
          ├─ Compute balanced class weights (sklearn)
          ├─ Phase 4: train frozen base, 120 epochs max, EarlyStopping(patience=25) + ReduceLROnPlateau
          ├─ Phase 5: unfreeze last 20 layers, fine-tune at lr=1e-5 (GPU required)
          └─ Evaluate: confusion matrix, accuracy, log-loss, per-class report, activation maps
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| EfficientNetB0 over custom CNN | Pre-trained ImageNet features transfer to retinal domain; outperforms a 4-block CNN from scratch with 5K images |
| Images at 224×224 (EfficientNetB0 native) | Native resolution maximises feature quality; requires ~7–8 GB RAM — Google Colab (T4 GPU) recommended |
| Validation extracted before augmentation | Prevents augmented copies of validation images from leaking into the training set |
| `float32 [0–255]` input (no `/255.0`) | EfficientNetB0 has a built-in rescaling layer; double normalisation degrades performance |
| Balanced class weights | Penalise errors on minority classes proportionally; prevents the model from collapsing to "predict Normal" |
| `model.evaluate()` replaced with sklearn | Keras 3 / TF 2.16+ bug in `optree` causes `ValueError` in `model.evaluate()` with `sparse_categorical_crossentropy` |

---

## Model Architecture

```
Input (224 × 224 × 3)
        │
┌───────▼──────────────────────────────────────────────┐
│  EfficientNetB0 (ImageNet pre-trained)               │
│  Output: (7 × 7 × 1280) feature maps                │
│  Phase 4: fully frozen (training=False)              │
│  Phase 5: last 20 layers unfrozen, lr = 1e-5         │
└───────┬──────────────────────────────────────────────┘
        │  GlobalAveragePooling2D  → (1280,)
        │  Dense(256, ReLU, L2=0.01)
        │  Dropout(0.5)
        │  Dense(8, softmax)
        ▼
  Class probabilities (8)
```

**Parameters:** 4,379,563 total | 329,992 trainable (Phase 4) | ~1.6M trainable (Phase 5, last 20 layers)

---

## Results

### Progression across training phases

| Phase | Change | Accuracy | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Baseline | Custom 4-block CNN from scratch | ~40% | ~0.30 | — |
| Phase 1 | Bug fixes (normalisation, validation, EarlyStopping) | ~45% | ~0.45 | — |
| Phase 2 | Balanced class weights | ~45% | 0.49 | — |
| Phase 4 | EfficientNetB0 frozen base @ 224×224 + tiered augmentation | 49.9% | 0.57 | 0.49 |
| **Phase 5** | **Fine-tuning (last 20 layers, lr=1e-5)** | **53.8%** | **0.56** | **0.52** |

> Phase 4 accuracy jumped from 37% to 50% when upgrading from 128×128 to 224×224 (EfficientNetB0 native resolution), confirming that input resolution is the dominant factor for frozen-base transfer learning on this dataset.

### Phase 5 — Per-class results (best model)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Normal | 0.67 | 0.58 | 0.62 | 575 |
| Diabetic Retinopathy | 0.48 | 0.45 | 0.46 | 322 |
| Glaucoma | 0.38 | 0.51 | 0.43 | 57 |
| Cataract | 0.81 | 0.81 | 0.81 | 59 |
| AMD | 0.45 | 0.62 | 0.52 | 53 |
| Hypertension | 0.14 | 0.16 | 0.15 | 25 |
| Myopia | 0.79 | 0.98 | 0.87 | 46 |
| Other | 0.27 | 0.35 | 0.31 | 142 |
| **Macro avg** | **0.50** | **0.56** | **0.52** | 1279 |

---

## Key Findings

1. **Global accuracy is misleading on imbalanced datasets.** A model that ignores Hypertension entirely
   can score higher accuracy than one that detects it. Macro recall and per-class F1 are the primary metrics.

2. **Transfer learning requires adequate input resolution.** EfficientNetB0 was designed for 224×224.
   Running at native resolution (~7–8 GB RAM) on a GPU instance is essential for full feature quality;
   lower resolutions measurably degrade performance.

3. **Resolution determines how much fine-tuning is needed.** At 128×128, Phase 4 scored only 37% — fine-tuning
   was required to recover +12 pp. At native 224×224, Phase 4 already reaches 50%, and Phase 5 adds only
   +4 pp. Higher-quality frozen features leave less for domain adaptation to correct.

4. **Class weights and augmentation interact.** Augmenting a class increases its sample count, which
   reduces its computed class weight. The two strategies must be calibrated jointly.

5. **Hypertension is statistically unreliable.** With 25 test samples, a difference of 10 predictions
   shifts recall by ±40 percentage points. Any metric reported for this class should be interpreted
   with caution.

6. **Augmentation order matters.** Generating augmented copies before extracting the validation set
   causes data leakage: augmented versions of validation images appear in the training set. This project
   corrects the order — validation is extracted first, augmentation is applied only to training images.

7. **Phase 5 shows moderate overfitting.** EarlyStopping triggers before the 100-epoch ceiling,
   but training accuracy still runs well ahead of validation accuracy. The gap is partly structural:
   the 22:1 class imbalance means augmented minority-class copies appear many times during training
   while the validation set reflects natural distribution.

---

## Limitations

- **Resolution:** 224×224 (EfficientNetB0 native). Requires ~7–8 GB RAM — local runs need a machine
  with sufficient memory; Google Colab (T4) is the recommended environment.
- **Hypertension class:** 103 training samples are insufficient for reliable learning. With 25 test
  samples, a shift of 10 predictions changes recall by ±40 pp — any metric for this class is
  statistically unreliable and should be interpreted with caution.
- **Overfitting in Phase 5:** Training accuracy exceeds test accuracy by a substantial margin.
  EarlyStopping (patience=20) triggers before the 100-epoch ceiling; Dropout(0.5) and L2
  regularisation are applied. Further improvement would require more training data or a
  lighter fine-tuning strategy (fewer unfrozen layers or lower learning rate).
- **Single-label assumption:** The ODIR-5K dataset contains multi-label patients (e.g. `DO`, `DH`).
  The pipeline uses the primary diagnosis only; multi-label classification is out of scope.
- **No external validation:** All evaluation is on the ODIR-5K test split. Generalisation to other
  fundus imaging devices or populations is untested.

---

## Project Structure

```
Ocular-Disease-Recognition/
├── README.md
├── FINAL_ANALYSIS.md                    # Results analysis, per-class breakdown, dataset challenges
├── requirements.txt
├── LICENSE
│
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb   # EDA, cleaning, train/test split
│   ├── 02_model_training.ipynb          # 8-class model, training (Phase 4 + 5), evaluation
│   └── 03_grouped_classification.ipynb  # 4-group clinical taxonomy classifier
│
├── utils/
│   └── image_utils.py                   # Shared load_images() and augment_images() helpers
│
├── figures/
│   ├── class_distribution.png           # Class count by gender
│   ├── class_imbalance.png              # Normal vs. all other conditions
│   ├── sample_images_per_class.png      # 10 fundus images per class
│   ├── augmented_samples.png            # Augmented image examples
│   └── group_distribution.png           # 8-class vs. 4-group distribution comparison
│
└── models/                              # Saved model weights (git-ignored)
    ├── efficientnetb0_phase4.keras               # 8-class frozen-base checkpoint
    ├── efficientnetb0_finetuned.keras            # 8-class fine-tuned checkpoint (best)
    ├── efficientnetb0_grouped_phase4.keras       # 4-group frozen-base checkpoint
    └── efficientnetb0_grouped_finetuned.keras    # 4-group fine-tuned checkpoint (best)
```

> `preprocessed_images/`, `Main/`, and `models/` are git-ignored.
> Download the dataset from Kaggle and place images in `preprocessed_images/` at the repository root.

---

## Quickstart

### Local

```bash
git clone https://github.com/JuanDiegoSalcedo/Ocular-Disease-Recognition.git
cd Ocular-Disease-Recognition

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download ODIR-5K from Kaggle and extract to preprocessed_images/
# Place train_data.h5 and test_data.h5 in Main/ (generated by notebook 1)

jupyter notebook notebooks/01_eda_and_preprocessing.ipynb
jupyter notebook notebooks/02_model_training.ipynb
```

### Google Colab

1. Upload `train_data.h5` and `test_data.h5` to `My Drive/ocular/`
2. Open `notebooks/02_model_training.ipynb` in Colab
3. Set **Runtime → Change runtime type → T4 GPU**
4. Uncomment and run the **Environment Setup** cell at the top
5. Run all remaining cells in order

---

## License

This project is distributed under the **GNU General Public License v2 (GPL-2.0)**.
See the `LICENSE` file for details.
