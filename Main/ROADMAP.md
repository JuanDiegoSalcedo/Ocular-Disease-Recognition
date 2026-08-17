# Roadmap de Implementación — Mejora Gradual de Precisión

## Contexto

El modelo base (`notebook_deep_learning_model.ipynb`) alcanzó **~40% de accuracy** en el conjunto de test con la arquitectura CNN original entrenada sobre el dataset ODIR-5K. Este documento describe el plan de mejora incremental: cada fase se implementa, se reentrena el modelo y se mide el impacto antes de avanzar a la siguiente.

El objetivo es llegar a **75–82% de accuracy** manteniendo buenos valores de recall en todas las clases, incluyendo las minoritarias (Glaucoma, AMD, Hipertensión).

---

## Estado actual

| Métrica | Valor |
|---|---|
| Accuracy global | ~40% |
| Recall — Normal (N) | 47% |
| Recall — Diabética (D) | 34% |
| Recall — Glaucoma (G) | 37% |
| Recall — Catarata (C) | 88% |
| Recall — AMD (A) | 19% |
| Recall — Hipertensión (H) | 8% |
| Recall — Miopía (M) | 83% |
| Recall — Otros (O) | 11% |

**Problemas identificados:**
- Píxeles no normalizados (red recibía valores `[0–255]` en lugar de `[0–1]`)
- `restore_best_weights=False` en EarlyStopping (el modelo guardado podía ser de una época mala)
- Datos augmentados incluidos en el set de validación (métricas de validación infladas)
- Red entrenada desde cero con ~5.000 imágenes y 8 clases desbalanceadas

---

## Fase 1 — Corrección de bugs (✅ Completada)

**Impacto esperado:** +5–15% accuracy → ~50–55%

### Cambios aplicados

#### 1.1 Normalización de píxeles
**Archivo:** `notebook_deep_learning_model.ipynb` — celda `load_images()`

Las redes neuronales convergen mucho más rápido y con mejor resultado cuando los valores de entrada están en el rango `[0.0, 1.0]`. El código original pasaba enteros `[0–255]` directamente.

```python
# Antes
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
images.append(img)

# Después
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype("float32") / 255.0   # ← normalización
images.append(img)
```

#### 1.2 Resize a 224×224
**Motivo:** las imágenes originales son 512×512. Cargar ~5.000 imágenes en esa resolución requiere ~16 GB de RAM. Reducir a 224×224 baja el consumo a ~1.5 GB y además es el tamaño estándar para transfer learning (Fase 4).

```python
IMAGE_SIZE = (224, 224)
img = cv2.resize(img, IMAGE_SIZE)
```

#### 1.3 `restore_best_weights=True`
Con `False`, si el entrenamiento empeoraba en las últimas épocas antes de que EarlyStopping lo detuviese, el modelo guardado era de la última época (no la mejor).

```python
EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
```

#### 1.4 Validación limpia
El código original construía un set de validación manual (`X_val`, `y_val`) pero lo ignoraba, usando en cambio `validation_split=0.2` sobre el dataset ya augmentado. Esto inflaba las métricas de validación porque imágenes artificiales entraban al set de evaluación.

```python
# Antes
history = model.fit(..., validation_split=0.2)

# Después
history = model.fit(..., validation_data=(X_val, y_val))
# X_val/y_val contiene solo imágenes originales, extraídas antes de la augmentación
```

---

## Fase 2 — Class Weights (pendiente)

**Impacto esperado:** accuracy global similar, pero **recall en clases minoritarias sube notablemente**

### Problema
El dataset está muy desbalanceado. El modelo aprende a "predecir Normal" porque minimiza la pérdida global aunque falle en las enfermedades raras.

| Clase | Muestras train | Muestras test |
|---|---|---|
| Normal (0) | ~2.300 | 575 |
| Diabética (1) | ~1.286 | 322 |
| Otros (7) | ~566 | 142 |
| Catarata (3) | ~234 | 59 |
| Glaucoma (2) | ~227 | 57 |
| AMD (4) | ~213 | 53 |
| Miopía (6) | ~186 | 46 |
| Hipertensión (5) | ~103 | 25 |

### Implementación
Agregar antes del entrenamiento:

```python
from sklearn.utils import class_weight
import numpy as np

class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_combined),
    y=y_train_combined
)
class_weights_dict = dict(enumerate(class_weights_array))
print("Class weights:", class_weights_dict)
```

Y activar el parámetro en `model.fit()`:

```python
history = model.fit(
    X_train_combined,
    y_train_combined,
    epochs=120,
    validation_data=(X_val, y_val),
    batch_size=32,
    callbacks=[early_stopping],
    class_weight=class_weights_dict   # ← activar
)
```

`compute_class_weight('balanced')` calcula automáticamente pesos inversamente proporcionales a la frecuencia de cada clase: Hipertensión recibirá ~22× más peso que Normal durante el cálculo de la pérdida.

---

## Fase 3 — Augmentación mejorada para clases pequeñas (pendiente)

**Impacto esperado:** +2–3% accuracy global, recall de Glaucoma/AMD/Hipertensión ↑

### Problema
La augmentación actual genera 3 copias por imagen para las clases no dominantes (flips + saturación), pero las clases más pequeñas (Glaucoma: 227, AMD: 213, Hipertensión: 103) siguen muy subrepresentadas.

### Implementación
Actualizar `augment_images()` para aplicar más variaciones a las clases con menos de 250 muestras (labels 2, 4, 5):

```python
def augment_images(images, labels):
    augmented_images = []
    augmented_labels = []

    for img, label in zip(images, labels):
        # Clases con pocas muestras: augmentación agresiva (6 copias)
        if label in [2, 4, 5]:
            variants = [
                tf.image.flip_left_right(img),
                tf.image.flip_up_down(img),
                tf.image.flip_left_right(tf.image.flip_up_down(img)),
                tf.image.random_brightness(img, max_delta=0.2),
                tf.image.rot90(img, k=1),
                tf.image.rot90(img, k=3),
            ]
            for v in variants:
                augmented_images.append(tf.image.random_saturation(v, 0.6, 1.4))
                augmented_labels.append(label)

        # Clases medianas: augmentación moderada (3 copias, igual que antes)
        elif label not in [0, 1]:
            h_flip = tf.image.flip_left_right(img)
            v_flip = tf.image.flip_up_down(img)
            hv_flip = tf.image.flip_left_right(v_flip)
            for v in [h_flip, v_flip, hv_flip]:
                augmented_images.append(tf.image.random_saturation(v, 0.6, 1.4))
                augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)
```

---

## Fase 4 — Transfer Learning con EfficientNetB0 (pendiente)

**Impacto esperado:** salto a **65–75% accuracy**

### Por qué funciona
EfficientNetB0 fue entrenado en ImageNet (14 millones de imágenes, 1.000 clases). Sus primeras capas ya aprendieron a detectar bordes, texturas y patrones visuales complejos — conocimiento que se transfiere directamente a imágenes de retina. En lugar de aprender desde cero con 5.000 imágenes, partimos de una base que ya "sabe ver".

### Implementación
Reemplazar la función `convolutional_model()` por:

```python
def build_transfer_model(input_shape=(224, 224, 3), num_classes=8):
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,       # sin la capa de clasificación original
        weights='imagenet'       # pesos preentrenados
    )
    base_model.trainable = False  # congelar: no modificar los pesos base

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

model = build_transfer_model()
model.summary()
```

El modelo se compila y entrena igual que en las fases anteriores. Como la base está congelada, solo se entrenan las capas densas superiores — rápido y con poco riesgo de overfitting.

> **Prerequisito:** las imágenes deben estar normalizadas a `[0–1]` (Fase 1). EfficientNetB0 aplica su propio preprocesamiento internamente.

---

## Fase 5 — Fine-tuning (pendiente)

**Impacto esperado:** +5–10% adicional → **75–82% accuracy**

### Qué es
Una vez que la cabeza densa convergió (Fase 4), se "descongelan" las últimas capas de EfficientNetB0 para que se adapten específicamente a imágenes de retina. Se usa un learning rate muy pequeño para no destruir el conocimiento preentrenado.

### Implementación
Ejecutar después de que el modelo de la Fase 4 haya convergido:

```python
# Descongelar el modelo base
base_model.trainable = True

# Mantener congeladas todas las capas excepto las últimas 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompilar con learning rate muy pequeño
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar con más paciencia (fine-tuning es lento)
fine_tuning_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

history_ft = model.fit(
    X_train_combined,
    y_train_combined,
    epochs=50,
    validation_data=(X_val, y_val),
    batch_size=16,        # batch más pequeño por mayor uso de memoria
    callbacks=[fine_tuning_stopping],
    class_weight=class_weights_dict
)
```

---

## Resumen de progresión esperada

```
Fase 0 (baseline)    ~40%  ████░░░░░░░░░░░░░░░░
Fase 1 (bugs)        ~53%  █████████░░░░░░░░░░░
Fase 2 (weights)     ~53%  █████████░░░░░░░░░░░  (mejora por clase, no global)
Fase 3 (augment)     ~56%  ██████████░░░░░░░░░░
Fase 4 (transfer)    ~72%  ██████████████░░░░░░
Fase 5 (fine-tune)   ~80%  ████████████████░░░░
```

> Los porcentajes son estimaciones. El salto real depende de la GPU disponible, el batch size y cuántas épocas se dejan correr.

---

## Cómo registrar el progreso

Después de cada fase, guardar la matriz de confusión en `output/` con nombre descriptivo:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Fase X — Accuracy: {accuracy:.2%}')
plt.savefig('output/faseX_descripcion.png', dpi=150, bbox_inches='tight')
plt.show()
```

Esto permite comparar visualmente cómo evoluciona cada clase entre fases.

---

## Prerequisitos de entorno

```bash
pip install pandas numpy matplotlib opencv-python scikit-learn tensorflow h5py tables
```

- Python 3.8+
- TensorFlow 2.x
- RAM mínima recomendada: 8 GB (16 GB para Fase 4+)
- GPU opcional pero recomendada para Fases 4 y 5 (sin GPU, cada época puede tardar 10–20 min)

---

## Fuente del dataset

**ODIR-5K** — Ocular Disease Intelligent Recognition  
Disponible en: `https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k`

Las imágenes deben colocarse en `preprocessed_images/` (un nivel arriba del notebook).
