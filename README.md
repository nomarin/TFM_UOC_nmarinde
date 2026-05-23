
## Trabajo final de master Ciencia de Datos, Universitat Oberta de Catalunya (UOC)
Desarrollo y evaluación de modelos de aprendizaje basados en los paradigmas nested learning ymodelos jerárquicos para la detección de cáncer de piel.

TFM Nora Marin de la Rosa

Modelo de deep learning para clasificar lesiones de piel del dataset HAM10000 en dos niveles simultáneos: benigno/maligno (nivel grueso) y entre las 7 patologías concretas (nivel fino). La arquitectura sigue el paradigma de Nested Learning, con tres optimizadores Adam independientes y una pérdida de consistencia jerárquica para penalizar contradicciones entre los dos niveles.

## Dataset

[HAM10000](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) — 10.000 imágenes dermatoscópicas de 7 clases:

| Código | Patología | Nivel |
|--------|-----------|-------|
| nv | Melanocytic nevi | benigno |
| bkl | Benign keratosis | benigno |
| df | Dermatofibroma | benigno |
| vasc | Vascular lesion | benigno |
| akiec | Actinic keratosis / Bowen | maligno |
| bcc | Basal cell carcinoma | maligno |
| mel | Melanoma | maligno |

La carpeta esperada es `./HAM10000_processed/` con la siguiente estructura:

```
HAM10000_processed/
├── train/
│   ├── nv/
│   ├── mel/
│   └── ...
├── val/
└── test/
```
---
## Arquitectura
El backbone es EfficientNetB0 preentrenado en ImageNet. Encima hay dos cabezas:

- **Cabeza gruesa**: Dense(128) → Dense(2, softmax) — predice benigno/maligno
- **Cabeza fina**: Dense(256) → Dense(7, softmax) — predice la patología concreta

La cabeza fina recibe concatenados los features del tronco más la salida de la cabeza gruesa (con `stop_gradient`). El stop_gradient es importante: sin él la pérdida de la cabeza fina intenta modificar también la cabeza gruesa, compitiendo con su propio gradiente y desestabilizando el entrenamiento.

La pérdida total es:

```
L = 0.3 · CE_gruesa + 1.0 · CE_fina + 0.1 · L_consistencia
```

`L_consistencia` penaliza los casos donde las probabilidades finas caen fuera de la superclase correcta (p.ej. el modelo predice melanoma pero asigna alta probabilidad a benigno en el nivel grueso).

---

## Entrenamiento

Dos fases:

**Warm-up (5 épocas)** — solo entrenan las cabezas, el backbone está congelado. Sirve para que las cabezas aprendan algo razonable antes de descongelar el backbone.

**Fine-tuning (15 épocas)** — se descongelan las últimas 50 capas del backbone. Las primeras capas se quedan congeladas para no perder los detectores básicos de ImageNet.

Los tres optimizadores Adam tienen cadencias de actualización distintas. El tronco se actualiza cada 4 pasos, la cabeza gruesa cada 4 pasos, la cabeza fina en cada paso. Esto sigue la lógica del paradigma NL: los niveles más abstractos cambian más despacio.

Para ejecutar:

```bash
python nestedmodel_V1_v2.1.py
```
---
## Requisitos
```
tensorflow >= 2.15
scikit-learn
pandas
matplotlib
seaborn
```
## Salidas

Todo se guarda en `./outputs_nested_v2.1/`:

- `best.weights.h5` — pesos del mejor checkpoint según `val_fine_acc`
- `training_log.csv` — métricas por época
- `confusion_coarse.png` / `confusion_fine.png` — matrices de confusión
- `final_metrics.csv` — accuracy, balanced accuracy, F1 macro y tasa de consistencia sobre el conjunto de test
- `classification_report.txt` — report completo de sklearn

---
