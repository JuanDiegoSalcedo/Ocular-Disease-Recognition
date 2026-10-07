# Final Analysis — Ocular Disease Recognition

## Overview

This document analyses the results of two deep learning experiments on the ODIR-5K retinal fundus
dataset: an 8-class fine-grained classifier (`02_model_training.ipynb`) and a 4-group clinical
taxonomy classifier (`03_grouped_classification.ipynb`). Both use EfficientNetB0 transfer learning
with a two-phase training strategy (frozen base + fine-tuning).

---

## Summary of Results

| Experiment | Phase 4 Accuracy | Phase 5 Accuracy | Phase 5 Macro F1 |
|---|---|---|---|
| 8-class (individual diagnoses) | 49.9% | 53.8% | 0.52 |
| 4-group (clinical taxonomy) | 59.3% | 61.5% | 0.60 |

The 4-group formulation achieves ~8 pp higher accuracy, but the improvement is partly structural
(fewer, more balanced classes) rather than purely a gain in discriminative power.

---

## Where the Model Performed Well

### Myopia — F1 0.87 (8-class)

The strongest result in the dataset. Pathological myopia produces distinctive morphological changes
visible in fundus photography: posterior pole stretching, a pale, elongated disc, and lacquer cracks.
These features are geometrically consistent across patients and contrast clearly with other conditions.
High recall (0.98) with precision (0.79) confirms genuine feature learning, not overprediction.

### Cataract — F1 0.81 (8-class)

Cataracts are unusual among the 8 classes because the condition directly degrades the fundus image
itself — a lens opacity scatters light, producing a characteristic hazy, low-contrast appearance.
The model effectively learned this image-quality signature rather than any specific retinal feature.
This is a tractable pattern even with relatively few training samples (234).

### Structural/Mechanical Group — F1 0.72 (4-group)

The 4-group experiment confirms that Glaucoma, Cataract, and Myopia share enough visual coherence
to be classified as a group with high reliability. All three involve structural changes to the
physical optics of the eye that manifest distinctly in fundus photography. This is the strongest
evidence in favour of the grouped formulation.

### AMD — F1 0.52 (8-class), contained in Degenerative group

Age-related Macular Degeneration produces characteristic drusen deposits in the macula — small,
bright yellow-white spots concentrated in the centre of the image. This spatial signature is
specific enough for the model to learn with only 213 training samples, particularly after tiered
augmentation (7 copies) brought its effective representation up to ~1,368 samples.

---

## Where the Model Struggled

### Hypertension — F1 0.15 (8-class)

The consistent worst performer across all experiments. The fundamental problem is data volume:
103 training samples after stratified splitting yield only ~83 images for training. Hypertensive
retinopathy produces subtle vascular changes (arteriovenous nicking, copper/silver wiring) that
require fine-grained feature extraction — features a model cannot reliably learn from 83 examples
regardless of augmentation strategy. The 9× augmentation generated 747 copies from the same 83
source images, producing near-duplicate patterns that do not meaningfully expand the feature space.

Recall (0.16) above precision (0.14) indicates the model slightly overpredicts this class — it
generates marginally more false positives than it misses positive cases, a pattern consistent with
the model being pushed by class weights to predict the minority class despite insufficient feature
evidence.

### Diabetic Retinopathy — F1 0.46 (8-class)

Counterintuitively, the second-largest class (1,286 training samples) does not rank among the
better-performing classes. Diabetic Retinopathy (DR) presents across a wide severity spectrum:
from mild microaneurismas to proliferative DR with neovascularisation. The ODIR-5K dataset does
not grade severity — all DR cases share a single label regardless of stage. This heterogeneity
within the class makes it harder to learn than more visually uniform conditions.

Additionally, DR co-occurs frequently with other conditions (DO, DH, DC combinations in the
dataset), and the pipeline collapses multi-label patients to a single primary diagnosis. This
introduces label noise: patients classified as "D" may have retinal features associated with their
secondary condition.

### Other diseases — F1 0.31 (8-class)

By construction, the "Other" class is a catch-all containing conditions not assigned to the
remaining 7 categories. It is the third-largest class (566 samples) but the most internally
heterogeneous. Without a consistent visual signature, the model cannot learn a reliable decision
boundary. The F1 of 0.31 reflects random-adjacent performance on a large, diverse class.

### Vascular/Metabolic Group — F1 0.53 (4-group)

Grouping Diabetic Retinopathy with Hypertension was clinically motivated (shared aetiology in
vascular damage) but visually inconsistent. DR has strong, identifiable features; Hypertension
does not. Within the group, the model effectively learns DR patterns and treats H as noise. The
F1 of 0.53 is slightly above the 8-class DR baseline (0.45), suggesting mild benefit from
absorbing Hypertension's small contribution into a larger group — but it does not solve the
underlying problem.

---

## Dataset Challenges

### 1. Severe class imbalance

The dataset has a 22:1 ratio between the largest class (Normal, 2,298 samples) and the smallest
(Hypertension, 103). Even after tiered augmentation, effective learning for the smallest classes
remains constrained by the limited diversity of source images. Class weighting partially compensates
but cannot substitute for genuine data diversity.

### 2. Multi-label patients collapsed to single label

A significant proportion of ODIR-5K patients carry co-occurring diagnoses (DO, DH, DC, DG, etc.).
The pipeline assigns a single primary label per image, discarding secondary conditions. This creates
label noise: an image labelled "D" may exhibit features of "O" or "H", and the model's predictions
for these images will be penalised even when they identify real secondary features. Multi-label
classification would be the correct formulation but is out of scope for this project.

### 3. Hypertension is fundamentally under-resourced

103 training samples represent a clinical reality, not a data engineering failure: hypertensive
retinopathy is less common and its fundus changes are subtle and highly variable. No augmentation
or weighting strategy compensates for this without external data.

### 4. Bilateral imaging with implicit patient-level dependency

Each patient contributes two images (left and right eye). The stratified split operates at the
image level, not the patient level. It is possible — though statistically limited — for the left
and right fundus of the same patient to appear in both training and test sets. This could provide
a marginal optimistic bias in evaluation metrics, particularly for patients with unusual features.

### 5. Small dataset for an 8-class medical imaging problem

5,113 training images across 8 classes yields an average of ~640 samples per class (median ~210
for minority classes). State-of-the-art retinal models are typically trained on datasets of 50,000+
images. The performance ceiling for this dataset with this architecture is limited primarily by
data volume, not model capacity.

### 6. Image quality variability

Fundus photography quality varies across imaging devices, operators, and patient cooperation.
The ODIR-5K dataset contains images from multiple clinical sources with inconsistent illumination,
focus, and field-of-view. This variability adds intra-class noise that the model must learn to
disregard.

---

## Technical Challenges Encountered

| Challenge | Root cause | Resolution |
|---|---|---|
| Data leakage in augmentation | Augmentation applied before val split | Restructured pipeline: val extracted first, augmentation on training only |
| EfficientNetB0 normalisation | Model has built-in rescaling; double normalisation degraded performance | Removed `/255.0` — images passed as `float32 [0–255]` |
| `model.evaluate()` crash | Keras 3 / TF 2.16+ `optree` bug with `sparse_categorical_crossentropy` | Replaced with `sklearn.metrics.accuracy_score` + `log_loss` |
| RAM constraint at 224×224 | ~7–8 GB required for full augmented training set | Google Colab T4 GPU required; 224×224 confirmed as necessary for full feature quality |
| Non-reproducible validation split | `np.random.choice` without seed | Fixed `np.random.seed(511)` before split |
| Non-reproducible augmentation | `tf.image.random_brightness` / `random_contrast` use TF's separate random state, not numpy | Added `tf.random.set_seed(511)` alongside `np.random.seed(511)` before split and augmentation |

---

## Key Takeaways

1. **Input resolution is the dominant factor for frozen-base transfer learning.** Moving from
   128×128 to 224×224 improved Phase 4 accuracy by +13 pp. The frozen base at native resolution
   extracts sufficiently rich features that fine-tuning adds only marginal gain (+4 pp).

2. **Class imbalance and augmentation must be calibrated jointly.** Augmenting a minority class
   increases its sample count, which reduces its computed class weight. Applying both independently
   risks either under-weighting (if augmentation is not accounted for in weight computation) or
   over-weighting (if augmentation is insufficient).

3. **Problem formulation matters as much as architecture.** The 4-group taxonomy achieves +6 pp
   macro F1 over 8 classes not because the model is better, but because Structural/Mechanical
   classes are genuinely more visually cohesive. Reformulating a problem around visual coherence
   is a valid modelling decision, but must be motivated by domain knowledge rather than metric
   optimisation.

4. **Some classes are not learnable with this data volume.** Hypertension (F1 0.13) and Other
   (F1 0.33) are not failures of architecture or training strategy — they are consequences of
   insufficient data diversity and heterogeneous labelling. Reporting these results transparently
   is more informative than excluding them.

5. **Macro metrics are the correct primary metric for imbalanced multi-class problems.**
   Global accuracy rewards the majority class. A model that always predicts Normal achieves ~45%
   accuracy on this dataset. Macro recall and per-class F1 expose whether minority classes are
   being learned at all.
