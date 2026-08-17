# Estructura y Conexiones del Proyecto

Este documento describe cómo se relacionan todos los componentes del proyecto entre sí. Complementa el `README.md` principal (qué hace el proyecto) y el `ROADMAP.md` (cómo mejorarlo).

---

## Mapa de archivos

```
Ocular-Disease-Recognition/
│
├── Main/                                  ← directorio principal del proyecto
│   ├── notebook_deep_learning_prep.ipynb  ← Notebook 1: limpieza y EDA
│   ├── notebook_deep_learning_model.ipynb ← Notebook 2: modelo y evaluación
│   │
│   ├── full_df.csv            ← datos crudos de entrada (ODIR-5K)
│   ├── preprocessed_data.csv  ← tabla limpia generada por Notebook 1
│   ├── train_data.h5          ← split de entrenamiento (generado por Notebook 1)
│   ├── test_data.h5           ← split de prueba     (generado por Notebook 1)
│   ├── dl_model.h5            ← modelo entrenado    (generado por Notebook 2)
│   │
│   ├── output/                ← capturas históricas de experimentos
│   │   ├── eda_*.png          ← visualizaciones exploratorias (Notebook 1)
│   │   └── exp_*.png          ← resultados de distintas versiones del modelo
│   │
│   ├── README.md              ← descripción general del proyecto
│   ├── ROADMAP.md             ← plan de mejora gradual de precisión
│   └── ARCHITECTURE.md        ← este archivo
│
└── preprocessed_images/       ← imágenes ODIR-5K (no incluidas en el repo)
    ├── 0_left.jpg
    ├── 0_right.jpg
    └── ...
```

> `preprocessed_images/` no se sube al repositorio (ver `.gitignore`).
> Descargarlo desde: `https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k`

---

## Flujo de datos

```
full_df.csv
    │
    │  notebook_deep_learning_prep.ipynb
    │  ├─ Limpia columnas redundantes
    │  ├─ Fusiona columnas de diagnóstico (N,D,G,C,A,H,M,O → 'diagnostic')
    │  ├─ Codifica target como entero (vector one-hot → índice)
    │  ├─ Filtra pacientes con un solo ojo
    │  ├─ Genera visualizaciones EDA → output/eda_*.png
    │  └─ Split estratificado 80/20
    │
    ├──► preprocessed_data.csv   (tabla completa limpia)
    ├──► train_data.h5           (80% — metadatos + nombres de imagen)
    └──► test_data.h5            (20% — metadatos + nombres de imagen)
                │
                │  notebook_deep_learning_model.ipynb
                │  ├─ Lee train_data.h5 / test_data.h5
                │  ├─ load_images(): carga imágenes desde preprocessed_images/
                │  │                 redimensiona a 224×224
                │  │                 normaliza a [0, 1]
                │  ├─ augment_images(): genera copias aumentadas de clases minoritarias
                │  ├─ Separa X_val/y_val (validación limpia, solo imágenes originales)
                │  ├─ Combina original + aumentado → X_train_combined
                │  ├─ Entrena CNN con EarlyStopping y validation_data
                │  └─ Evalúa: matriz de confusión, accuracy, classification_report
                │
                └──► dl_model.h5   (modelo entrenado, listo para inferencia)
```

---

## Relación entre los notebooks

Los notebooks deben ejecutarse **en orden**. El segundo depende de las salidas del primero:

| Notebook 1 genera | Lo usa Notebook 2 en |
|---|---|
| `train_data.h5` | Carga inicial de datos de entrenamiento |
| `test_data.h5` | Carga inicial de datos de prueba |
| Columna `filename` (en .h5) | `load_images()` para leer las imágenes |
| Columna `target` (entero 0–7) | Etiquetas para entrenamiento y evaluación |

> `preprocessed_data.csv` no lo usa Notebook 2 directamente — es solo una copia legible de la tabla limpia para inspección manual.

---

## Relación entre los archivos `.h5`

Los archivos `.h5` **no contienen imágenes**. Contienen una tabla con metadatos del paciente y el nombre del archivo de imagen:

```
train_data.h5 / test_data.h5
│
├── ID              → identificador del paciente
├── Patient Age     → edad
├── Patient Sex     → sexo
├── target          → clase como entero (0=Normal, 1=Diabética, ..., 7=Otros)
├── filename        → nombre del archivo JPG en preprocessed_images/
├── diagnostic      → string de diagnóstico (ej. "GA" = Glaucoma + AMD)
└── binary_target   → 1 si Normal, 0 si cualquier enfermedad
```

La relación entre `filename` y la imagen real es:

```python
img_path = os.path.join('../preprocessed_images', row['filename'])
# Ejemplo: ../preprocessed_images/4584_left.jpg
```

---

## Clases del modelo

| Índice (target) | Código | Enfermedad |
|---|---|---|
| 0 | N | Normal |
| 1 | D | Retinopatía Diabética |
| 2 | G | Glaucoma |
| 3 | C | Catarata |
| 4 | A | Degeneración Macular Asociada a la Edad (AMD) |
| 5 | H | Hipertensión |
| 6 | M | Miopía Patológica |
| 7 | O | Otras enfermedades |

---

## Arquitectura de la CNN

```
Imagen de entrada (224 × 224 × 3)
        │
┌───────▼────────────────────────────────────────┐
│  Bloque 1: Conv2D(16) → BN → ReLU             │
│            MaxPool(2×2) → Dropout(0.1)         │  → (112 × 112 × 16)
├────────────────────────────────────────────────┤
│  Bloque 2: Conv2D(32) → BN → ReLU             │
│            MaxPool(2×2) → Dropout(0.1)         │  → (56 × 56 × 32)
├────────────────────────────────────────────────┤
│  Bloque 3: Conv2D(32) → BN → ReLU             │
│            MaxPool(2×2) → Dropout(0.1)         │  → (28 × 28 × 32)
├────────────────────────────────────────────────┤
│  Bloque 4: Conv2D(16) → BN → ReLU             │
│            MaxPool(2×2) → Dropout(0.1)         │  → (14 × 14 × 16)
├────────────────────────────────────────────────┤
│  Flatten → Dense(256, ReLU) → Dropout(0.4)    │  → (256,)
├────────────────────────────────────────────────┤
│  Dense(8, Softmax)                             │  → probabilidades por clase
└────────────────────────────────────────────────┘
```

Cada bloque aplica regularización L2 (0.01) en Conv2D y Dense para reducir overfitting.

---

## Carpeta `output/` — convenciones de nombres

Los archivos están organizados con prefijos para indicar su origen:

| Prefijo | Origen | Descripción |
|---|---|---|
| `eda_` | Notebook 1 | Visualizaciones exploratorias del dataset |
| `exp_` | Notebook 2 | Resultados de experimentos del modelo |

### Archivos EDA
| Archivo | Contenido |
|---|---|
| `eda_distribucion_clases.png` | Conteo de casos por enfermedad |
| `eda_desbalance_clases.png` | Comparación Normal vs. resto |
| `eda_ejemplos_imagenes_reales.png` | Grilla de 10 imágenes por clase |
| `eda_ejemplos_imagenes_aumentadas.png` | Ejemplos de imágenes generadas por augmentación |

### Archivos de experimentos (evolución del modelo)
| Archivo | Accuracy | Descripción |
|---|---|---|
| `exp_22pct_acc_con_augmentacion.png` | ~22% | Primer intento con augmentación |
| `exp_30pct_acc_dropout01.png` | ~30% | Dropout 0.1 |
| `exp_30pct_acc_augmentado.png` | ~30% | Con augmentación |
| `exp_33pct_acc_augmentado.png` | ~33% | Augmentación mejorada |
| `exp_37pct_acc_conv.png` | ~37% | Modelo convolucional |
| `exp_3capas_red_densa.png` | — | Experimento con red densa pura |
| `exp_3capas_cnn_baseline.png` | ~41% | CNN con 3 capas |
| `exp_40pct_acc_regularizacion_dropout.png` | ~40% | Regularización + dropout |
| `exp_41pct_acc.png` | ~41% | Iteración 4 capas |
| `exp_43pct_acc_4capas_mejor_resultado.png` | ~43% | **Mejor resultado registrado** |
| `exp_baseline_confusion_matrix.png` | — | Matriz de confusión baseline |
| `exp_mas_epocas_curvas_entrenamiento.png` | — | Efecto de más épocas |

---

## Dependencias entre componentes (resumen)

```
full_df.csv
    └─► [Notebook 1] ─┬─► train_data.h5 ──► [Notebook 2] ─► dl_model.h5
                      ├─► test_data.h5  ──► [Notebook 2]
                      └─► preprocessed_data.csv

preprocessed_images/
    └─► [Notebook 2: load_images()] ─► X_train_images / X_test_images

dl_model.h5
    └─► inferencia futura (cargar con tf.keras.models.load_model('dl_model.h5'))
```
