# C3 — Ocular Disease Recognition with Deep Learning

A deep-learning project that classifies **ocular (eye) diseases** from retinal
fundus images using the Kaggle **ODIR-5K** dataset. The pipeline cleans and
pre-processes the tabular metadata, trains a custom **Convolutional Neural
Network (CNN)**, augments the underrepresented classes, and evaluates the model
with confusion matrices, accuracy, per-class reports, and layer-activation
visualizations.

The project is organized as two Jupyter notebooks that must be run in order:

| Notebook | Purpose |
| --- | --- |
| `notebook_deep_learning_prep.ipynb` | Exploratory analysis, data cleaning, preprocessing and train/test split |
| `notebook_deep_learning_model.ipynb` | CNN construction, data augmentation, training and evaluation |

---

## Problem & Classes

The model performs **multiclass classification** over **8 categories** of eye
condition:

| Code | Class |
| --- | --- |
| `N` | Normal |
| `D` | Diabetic Retinopathy |
| `G` | Glaucoma |
| `C` | Cataract |
| `A` | Age-related Macular Degeneration |
| `H` | Hypertension |
| `M` | Pathological Myopia |
| `O` | Other diseases / abnormalities |

---

## Core Components

### 1. Data preprocessing notebook (`notebook_deep_learning_prep.ipynb`)

- Loads the raw `full_df.csv` (ODIR-5K metadata + eye image paths).
- **Cleaning**:
  - Drops redundant columns: `Left-Fundus`, `Right-Fundus`,
    `Left-Diagnostic Keywords`, `Right-Diagnostic Keywords`, `filepath`, and
    the individual one-hot diagnosis columns.
  - Merges the one-hot columns `N, D, G, C, A, H, M, O` into a single
    `diagnostic` string column.
- **Target encoding**:
  - Converts the stringified one-hot `target` into an **integer class index**
    (via `list.index(1)`).
  - Adds `binary_target` (1 if Normal, 0 otherwise) for the healthy-vs-disease
    analysis.
- **Filtering**: keeps patients whose `ID` has both eyes (`ID` count == 2),
  dropping patients with photos of only one eye.
- **Exploratory analysis / visualizations**:
  - Class distribution and imbalance inspection.
  - Target distribution by gender and sex (pie + grouped bar charts).
  - Healthy vs. affected relationship.
  - Age distribution (peak ~60 years).
  - Sample image grid (10 per class), both eyes of a random patient,
    image-channel inspection.
  - Confirms all images are `512 x 512 x 3`.
- **Train/test split**: stratified 80 / 20 split using `random_state=511`.
- **Saves outputs**: `train_data.h5`, `test_data.h5` (HDF5), and
  `preprocessed_data.csv`.

### 2. Model notebook (`notebook_deep_learning_model.ipynb`)

- **Loads** the previously generated `train_data.h5` / `test_data.h5`.
- **`load_images()`**: reads each image file, converts **BGR → RGB**
  (OpenCV), and returns a NumPy array of images.
- **CNN construction** (see architecture below).
- **Data augmentation — `augment_images()`**: for underrepresented classes,
  generates augmented copies using horizontal/vertical flips and random
  saturation tweaks (`lower=0.6`, `upper=1.4`) to balance class sizes.
- **Manual validation split**: reserves 1100 samples from class `0` and 300
  from class `1` for validation.
- **Combine & shuffle**: concatenates original + augmented training data and
  shuffles the resulting set.
- **Training**: compiles with `adam` + `sparse_categorical_crossentropy`,
  monitors `accuracy`, uses `EarlyStopping` on `val_loss` (patience 100), and
  trains for up to 120 epochs with `validation_split=0.2`, `batch_size=32`.
- **Evaluation**:
  - Confusion matrix (`ConfusionMatrixDisplay`).
  - Loss / accuracy on the test set.
  - Per-class `classification_report`.
  - **Layer-activation visualization** on a random test image to inspect what
    each convolutional layer has learned.
- **Saves** the trained model to `dl_model.h5`.
---

## CNN Architecture

The `convolutional_model(input_shape=(512, 512, 3))` builds the following stack:

| # | Layer | Details |
| --- | --- | --- |
| 1 | Conv2D | 16 filters, `3x3`, `padding='same'`, `he_normal`, L2 (0.01) |
| 2 | BatchNormalization → ReLU | |
| 3 | MaxPooling2D | `2x2`, stride 2 |
| 4 | Dropout | 0.1 |
| 5 | Conv2D | 32 filters, `3x3`, `padding='same'`, `he_normal`, L2 (0.01) |
| 6 | BatchNormalization → ReLU | |
| 7 | MaxPooling2D | `2x2`, stride 2 |
| 8 | Dropout | 0.1 |
| 9 | Conv2D | 32 filters, `3x3`, `padding='same'`, `he_normal`, L2 (0.01) |
| 10 | BatchNormalization → ReLU | |
| 11 | MaxPooling2D | `2x2`, stride 2 |
| 12 | Dropout | 0.1 |
| 13 | Conv2D | 16 filters, `3x3`, `padding='same'`, `he_normal`, L2 (0.01) |
| 14 | BatchNormalization → ReLU | |
| 15 | MaxPooling2D | `2x2`, stride 2 |
| 16 | Dropout | 0.1 |
| 17 | Flatten | |
| 18 | Dense | 256 units, ReLU, `he_normal`, L2 (0.01) |
| 19 | Dropout | 0.4 |
| 20 | Dense (output) | 8 units, `softmax`, L2 (0.01) |

Every Conv/Dense layer uses **L2 regularization** (`0.01`) to reduce overfitting.

---

## Pipeline (Main Code Logic)

```
full_df.csv
  └─(prep notebook)─┐
     drop redundant columns
     merge one-hot N,D,G,C,A,H,M,O → diagnostic
     encode target → integer class index
     keep patients with both eyes
     EDA / visualizations
     stratified 80/20 split
     └─► train_data.h5 / test_data.h5 / preprocessed_data.csv
                       │
(train_data.h5)  ┌────┘   (test_data.h5)
                 │
    load_images() (BGR→RGB)
                 │
    CNN: Conv2D×4 → BN/ReLU → MaxPool → Drop → Flatten → Dense(256) → Dense(8)
                 │
    augment_images() (flips + saturation) for underrepresented classes
                 │
    combine original + augmented, shuffle
                 │
    compile (adam, sparse_categorical_crossentropy) + EarlyStopping
                 │
    fit → epochs=120, validation_split=0.2, batch_size=32
                 │
    ┌──── model.save('dl_model.h5')
    │
    evaluate: confusion matrix, accuracy, classification_report,
    activation maps
```

---

## Requirements

Install the dependencies used by the notebooks:

```bash
pip install pandas numpy matplotlib opencv-python scikit-learn tensorflow h5py
```

- Python 3.x
- TensorFlow / Keras
- OpenCV (`cv2`)
- scikit-learn
- HDF5 support (for `read_hdf` / `to_hdf`)

> **Note**: the notebooks reference Kaggle paths
> (`/kaggle/input/...`). Adjust the data / image folder paths to your local
> environment before running.

---

## Usage

1. Run `notebook_deep_learning_prep.ipynb` first to generate
   `train_data.h5`, `test_data.h5`, and `preprocessed_data.csv`.
2. Run `notebook_deep_learning_model.ipynb` to build, train, and evaluate the
   model, producing `dl_model.h5`.
3. Inspect the resulting plots (confusion matrix, activations, augmented
   samples) and the evaluation metrics.

---

## Output Artifacts

| File | Description |
| --- | --- |
| `train_data.h5` / `test_data.h5` | Preprocessed train/test splits (HDF5) |
| `preprocessed_data.csv` | Cleaned tabular dataset |
| `dl_model.h5` | Trained Keras model |
| `output/*.png` | Confusion matrices, accuracy curves, augmentation/activation plots, EDA figures |

---

## Data Source

- **ODIR-5K** — Ocular Disease Intelligent Recognition (retinal fundus images
  and patient metadata).

## License

This project is distributed under the **GNU General Public License v2 (GPL-2.0)**.
See the `LICENSE` file for details.
