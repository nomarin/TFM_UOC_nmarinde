"""nestedmodel_V1_v2.0 — Modelo jerárquico con Nested Optimizers y perdida de consistencia

Este codigo cooresponde a la implementación final del modelo de clasificación jerárquica propuesto para el TFM.

Salidas
-------
Genera dentro de ``./outputs_nested_v2/``:
  - ``best.weights.h5``        — pesos del mejor checkpoint según ``val_fine_acc``.
  - ``training_log.csv``       — histórico métricas por época (CSVLogger).
  - ``confusion_coarse.png``   — matriz de confusión nivel grueso (post-eval).
  - ``confusion_fine.png``     — matriz de confusión nivel fino (post-eval).

Ejemplo de uso
$ python nestedmodel_V1_v2.0.py
"""

# PREPARACIÓN DEL ENTORNO Y LIBRERÍAS

# En este bloque importamos todas las librerias necesarias para la ejecución del código. Además se silencian algunas de las alertas del sistema de los logs de TensorFlow para que sea mas sencillo de leer los resultados

# Importamos también la arquitectura preentrenda de EfficientNetB0, que se usará como backbone del modelo jerárquico 

# La función preprocess_input se encargará de escalar las imágenes al rango [-1, 1] usando la media y desviación estándar de ImageNet

# configuración del entorno para evitar logs de TensorFlow

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]= "2"

# librerías estándar
import random
import numpy as np
import pandas as pd
import tensorflow as tf

# keras para construcción de modelos, optimizadores y callbacks
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

# métricas de clasificación y visualización, incluyendo balanced accuracy y F1 macro para evaluar el rendimiento en un dataset desbalanceado (como HAM10000)
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, f1_score)

import matplotlib.pyplot as plt
import seaborn as sns

# fijamos las semillas 

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# RUTAS Y CONFIGURACIÓN

# ruta destino para las salidas del modelo
DATA_DIR = "./HAM10000_processed"

# directorio para guardar los checkpoints, logs y figuras de esta versión del modelo. Se crea si no existe.
OUTPUT_DIR = "./outputs_nested_v2.0"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# HIPERPARÁMETROS

# tamaño de las imagenes de entrada y el tamño de batch para el entrenamiento. 
# han sido definidos en base a la arquitectura EfficientNetB0. 
IMG_SIZE = 224
BATCH_SIZE = 32

# tenemos dos fases de entrenamiento, definimos para cada una:

# el numero de epocas
# congelamos el extractor de las características y posteriormente lo descongelamos para el fine-tuning. 
EPOCHS_WARMUP = 5
EPOCHS_FINETUNE = 15

# tres tasas de aprendizaje
LR_FAST = 1e-4   # cabeza fina
LR_MID = 5e-5   # cabeza gruesa
LR_SLOW = 5e-5   # tronco

# cadencias de actualización - asíncronas para cada nivel del modelo jerárquico. 
CAD_FAST = 1
CAD_MID = 4
CAD_SLOW = 4

# pesos de la perdida compuesta
# beta más alto ya que la clasificacion fina es el objetivo principal
# gamma bajo para no penalizr demasiado la consistencia jerárquica
ALPHA_COARSE  = 0.3
BETA_FINE     = 1.0
GAMMA_CONSIST = 0.1
LABEL_SMOOTH  = 0.05

# definicion de la estructura jerarquica

# las clases del conjunto de datos (HAM10000) estan definidas con la siguiente nomeclatura:
# akiec - actinic keratosis / Bowen (precanceroso)
# bcc - basal cell carcinoma (maligno)
# bkl - benign keratosis (benigno)
# df - dermatofibroma (benigno)
# mel - melanoma (maligno)
# nv - melanocytic nevi (benigno)
# vasc - vascular lesion (benigno)

# definicion las clases y el criterio clinico (gruesas).
FINE_CLASSES   = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
FINE_TO_IDX    = {c: i for i, c in enumerate(FINE_CLASSES)}
COARSE_CLASSES = ["benign", "malignant"]

# mapeo de las clases finas a las clases gruesas. Las benignas se asignan a 0 y las malignas (incluyendo la precancerosa) a 1
FINE_TO_COARSE = {"nv":0,"bkl":0,"df":0,"vasc":0, # benignas = O
                  "akiec":1,"bcc":1,"mel":1}  # malignas (mas precancerosa)=1

# se crea un matriz de ceros donde cada columna le diga la modelo a que clase gruesa pertenece cada enfermedad (fina)
HIERARCHY_MATRIX = np.zeros((2, 7), dtype=np.float32)
for f, i in FINE_TO_IDX.items():
    HIERARCHY_MATRIX[FINE_TO_COARSE[f], i] = 1.0

# convertimos a tensor de TensorFlow para usar en el modelo durante el entrenamiento.
HIERARCHY_MATRIX_TF = tf.constant(HIERARCHY_MATRIX)


# CARGAMOS LOS DATOS
def load_split(split):
    """Carga las imágenes de un split (train/val/test) y devuelve un dataframe
    con la ruta de cada imagen y sus etiquetas fina y gruesa."""
    rows = []
    for dx in FINE_CLASSES:
        carpeta = os.path.join(DATA_DIR, split, dx)
        if not os.path.exists(carpeta):
            continue   # puede que alguna clase no tenga carpeta en val/test
        for nombre in os.listdir(carpeta):
            if not nombre.endswith(".jpg"):
                continue
            rows.append({
                "path":os.path.join(carpeta, nombre),
                "dx":dx,
                "fine":FINE_TO_IDX[dx],
                "coarse":FINE_TO_COARSE[dx]
            })
    return pd.DataFrame(rows)
 
 
def make_ds(df, training):
    """Prepara un tf.data.Dataset a partir de un dataframe con rutas y etiquetas."""
    # preparamos los datos iniciales, converimos las etiquetas en on-hot encoding (vectores de O y 1).
    paths = df["path"].values
    yf = tf.keras.utils.to_categorical(df["fine"].values, num_classes=7)
    yc = tf.keras.utils.to_categorical(df["coarse"].values, num_classes=2)
 
    ds = tf.data.Dataset.from_tensor_slices((paths, yc, yf))
 
    if training:
        # mezclo antes de leer las imágenes para no tener que cargarlas todas en RAM
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
 
    def cargar_imagen(ruta, etiq_gruesa, etiq_fina):
        img = tf.io.read_file(ruta)
        img = tf.io.decode_jpeg(img, channels=3)
        # normalizamos con preprocess_input de efficientnet escala al rango [-1, 1]
        img = preprocess_input(tf.cast(img, tf.float32))
        return img, {"coarse_out": etiq_gruesa, "fine_out": etiq_fina}

    # utilizamos map para cargar y procesar las imágenes de forma paralela, luego batch y prefetch para optimizar el pipeline de datos durante el entrenamiento.
    return (ds.map(cargar_imagen, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(BATCH_SIZE)
              .prefetch(tf.data.AUTOTUNE))

# ARQUITECTURA

def build_model(trainable_backbone=True):
    """
    Definimos la arquitectura del modelo (llamado HierSkin - Hierarchical Skin)
    Modelo jerárquico con dos cabezas de clasificación:
      - cabeza gruesa: predice benigno/maligno
      - cabeza fina: predice la patología específica entre 7 clases
 
    La cabeza fina recibe también la salida de la cabeza gruesa (top-down) de lo general a lo especifico, con esto conseguimos que la cabeza fina tenga una guía
    así sabe "qué tipo de lesión espera ver" antes de clasificar.
 
    Uso stop_gradient para que la pérdida fina no modifique los pesos
    de la cabeza gruesa. Sin esto, los dos objetivos compiten y el
    entrenamiento se vuelve inestable (lo comprobé en versiones anteriores).
    """

    # cargamos el backbone preentrenado de EfficientNetB0 sin la cabeza de clasificación (include_top=False) y con pesos de ImageNet
    backbone = EfficientNetB0(
        include_top=False, # quitamos la cabeza de ImageNet
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    backbone.trainable = trainable_backbone
 
    entrada = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
 
    # extraemos características con el backbone
    # hay que pasar el mismo valor de training que tiene el backbone,
    # si no las BatchNorm internas se comportan mal durante el warm-up
    x = backbone(entrada, training=trainable_backbone)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="trunk_dropout")(x)
 
    # cabeza gruesa: decide si la lesión es benigna o maligna

    c = layers.Dense(128, activation="relu", name="coarse_dense")(x)
    c = layers.Dropout(0.3, name="coarse_dropout")(c)
    salida_gruesa = layers.Dense(2, activation="softmax", name="coarse_out")(c)
 
    # cabeza fina: usa los features del tronco + la decisión gruesa
    # stop_gradient: la cabeza fina VE la salida gruesa pero NO puede modificarla
    gruesa_detenida = tf.keras.ops.stop_gradient(salida_gruesa)
    combinado = layers.Concatenate(name="nest_concat")([x, gruesa_detenida])
    f = layers.Dense(256, activation="relu", name="fine_dense")(combinado)
    f = layers.Dropout(0.4, name="fine_dropout")(f)
    salida_fina = layers.Dense(7, activation="softmax", name="fine_out")(f)
 
    return models.Model(
        entrada,
        {"coarse_out": salida_gruesa, "fine_out": salida_fina},
        name="HierSkin"
    )
 
# PÉRDIDA DE CONSISTENCIA JERÁRQUICA
 
class HierConsistencyLoss(tf.keras.losses.Loss):
    """
    Penaliza casos donde la cabeza fina "contradice" a la cabeza gruesa.
    Por ejemplo: predecir melanoma (maligno) pero con alta probabilidad en benigno a nivel grueso
    """
    def __init__(self, M, name="hier_consist"):
        super().__init__(name=name)
        self.M = M  # matriz de herencia (2 x 7)
 
    def call(self, yc_real, yf_pred):
        # multiplicamos la etiqueta gruesa one-hot por la matriz de herencia
        # para obtener una máscara de qué clases finas son "válidas"
        mascara = tf.matmul(yc_real, self.M)
        prob_en_superclase = tf.reduce_sum(yf_pred * mascara, axis=-1)
    
        return -tf.math.log(prob_en_superclase + 1e-7)
 
 
# MODELO NESTED CON GRADIENT MASKING
 
class NestedHierModel(tf.keras.Model):
    """
    Wrapper que implementa los nested optimizers del paradigma Nested Learning.
 
    La idea es tener tres optimizadores Adam independientes (uno por nivel:
    tronco, cabeza gruesa, cabeza fina) con cadencias de actualización distintas. Así el tronco cambia más despacio que las cabezas.
    """
 
    def __init__(self, base_model, hierarchy_matrix, **kwargs):
        super().__init__(**kwargs)
 
        self.base  = base_model
        self.alpha = ALPHA_COARSE
        self.beta  = BETA_FINE
        self.gamma = GAMMA_CONSIST
 
        # pérdidas para cada nivel
        self.ce_gruesa = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)
        self.ce_fina   = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)
        self.consist   = HierConsistencyLoss(hierarchy_matrix)
 
        # un optimizador adam por nivel — cada uno guarda sus propios momentos
        # eso es lo que el paradigma NL llama "memorias asociativas independientes"
        self.opt_slow = optimizers.Adam(learning_rate=LR_SLOW)   # tronco
        self.opt_mid  = optimizers.Adam(learning_rate=LR_MID)    # cabeza gruesa
        self.opt_fast = optimizers.Adam(learning_rate=LR_FAST)   # cabeza fina
 
        # cadencias — cad_slow es Variable para poder cambiarla entre fases
        self.cad_slow = tf.Variable(CAD_SLOW, dtype=tf.int64, trainable=False)
        self.cad_mid  = tf.constant(CAD_MID,  dtype=tf.int64)
        self.cad_fast = tf.constant(CAD_FAST, dtype=tf.int64)
 
        # contador de pasos — tiene que ser tf.Variable para sobrevivir al trazado de @tf.function
        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
 
        # métricas que quiero ver durante el entrenamiento
        self.m_loss = tf.keras.metrics.Mean(name="loss")
        self.m_c    = tf.keras.metrics.CategoricalAccuracy(name="coarse_acc")
        self.m_f    = tf.keras.metrics.CategoricalAccuracy(name="fine_acc")
        self.m_cn   = tf.keras.metrics.Mean(name="consist")
 
    @property
    def metrics(self):
        # hay que sobreescribir esto o keras no resetea las métricas entre épocas
        return [self.m_loss, self.m_c, self.m_f, self.m_cn]
 
    def call(self, inputs, training=False):
        return self.base(inputs, training=training)
 
    def _split(self):
        """Separa las variables entrenables en tres grupos según el nivel."""
        tronco = self.base.get_layer("efficientnetb0").trainable_variables
        gruesa = (self.base.get_layer("coarse_dense").trainable_variables +
                  self.base.get_layer("coarse_out").trainable_variables)
        fina   = (self.base.get_layer("fine_dense").trainable_variables +
                  self.base.get_layer("fine_out").trainable_variables)
        return tronco, gruesa, fina
 
    def _loss(self, y, yp):
        """Pérdida total = alpha*CE_gruesa + beta*CE_fina + gamma*consistencia"""
        yc_real, yf_real = y["coarse_out"], y["fine_out"]
        yc_pred, yf_pred = yp["coarse_out"], yp["fine_out"]
 
        l_gruesa = self.ce_gruesa(yc_real, yc_pred)
        l_fina   = self.ce_fina(yf_real, yf_pred)
        # HierConsistencyLoss devuelve un tensor (batch,) así que hay que reducirlo
        l_cons   = tf.reduce_mean(self.consist(yc_real, yf_pred))
 
        total = self.alpha * l_gruesa + self.beta * l_fina + self.gamma * l_cons
        return total, l_gruesa, l_fina, l_cons
 
    def train_step(self, data):
        x, y = data
        tronco_v, gruesa_v, fina_v = self._split()
        todas_vars = tronco_v + gruesa_v + fina_v
 
        # un solo backward para los tres niveles — más eficiente que tres separados
        with tf.GradientTape() as tape:
            yp = self(x, training=True)
            total, lc, lf, lcons = self._loss(y, yp)
        grads = tape.gradient(total, todas_vars)
 
        # separacion de los gradientes en el mismo orden en que concatené las variables
        n_t = len(tronco_v)
        n_c = len(gruesa_v)
        g_tronco = grads[:n_t]
        g_gruesa = grads[n_t:n_t + n_c]
        g_fina   = grads[n_t + n_c:]
 
        paso = self.global_step
 
        # las máscaras controlan si toca actualizar en este paso o no
        # si mask=0 el gradiente se anula y adam no modifica los pesos ese paso
        mask_fina   = tf.cast(tf.equal(paso % self.cad_fast, 0), tf.float32)
        mask_gruesa = tf.cast(tf.equal(paso % self.cad_mid,  0), tf.float32)
        mask_tronco = tf.cast(tf.equal(paso % self.cad_slow, 0), tf.float32)
 
        if len(fina_v) > 0:
            self.opt_fast.apply_gradients(
                zip([g * mask_fina for g in g_fina], fina_v))
        if len(gruesa_v) > 0:
            self.opt_mid.apply_gradients(
                zip([g * mask_gruesa for g in g_gruesa], gruesa_v))
        if len(tronco_v) > 0:
            self.opt_slow.apply_gradients(
                zip([g * mask_tronco for g in g_tronco], tronco_v))
 
        self.global_step.assign_add(1)
 
        self.m_loss.update_state(total)
        self.m_c.update_state(y["coarse_out"], yp["coarse_out"])
        self.m_f.update_state(y["fine_out"],   yp["fine_out"])
        self.m_cn.update_state(lcons)
        return {m.name: m.result() for m in self.metrics}
 
    def test_step(self, data):
        """Igual que train_step pero sin actualizar pesos ni el contador"""
        x, y = data
        yp = self(x, training=False)
        total, lc, lf, lcons = self._loss(y, yp)
        self.m_loss.update_state(total)
        self.m_c.update_state(y["coarse_out"], yp["coarse_out"])
        self.m_f.update_state(y["fine_out"],   yp["fine_out"])
        self.m_cn.update_state(lcons)
        return {m.name: m.result() for m in self.metrics}
 
# PIPELINE PRINCIPAL
 
# imprime los hiperparámetros al inicio para saber qué configuración estoy corriendo
print(f"LR_FAST={LR_FAST}  LR_MID={LR_MID}  LR_SLOW={LR_SLOW}")
print(f"CAD={CAD_FAST}:{CAD_MID}:{CAD_SLOW}")
print(f"ALPHA={ALPHA_COARSE}  BETA={BETA_FINE}  GAMMA={GAMMA_CONSIST}  LS={LABEL_SMOOTH}")
 
# carga de datos
train_df = load_split("train")
val_df   = load_split("val")
test_df  = load_split("test")
print(f"\nTrain: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
 
# distribución de clases para confirmar el desbalance
print("\nDistribución de clases en train:")
print(train_df["dx"].value_counts().to_string())
 
train_ds = make_ds(train_df, training=True)
val_ds   = make_ds(val_df,   training=False)
test_ds  = make_ds(test_df,  training=False)
 
# construcción del modelo
base   = build_model(trainable_backbone=True)
hmodel = NestedHierModel(base, HIERARCHY_MATRIX_TF)
# keras exige un optimizer en compile aunque usemos el train_step custom
# pongo un adam de placeholder para que no de error
hmodel.compile(optimizer=optimizers.Adam())
 
# callbacks
ckpt_path = os.path.join(OUTPUT_DIR, "best.weights.h5")
cbs = [
    # patience=10 porque el tronco tarda varias épocas en asentarse
    callbacks.EarlyStopping(
        monitor="val_fine_acc", mode="max",
        patience=10, restore_best_weights=True, verbose=1
    ),
    callbacks.ModelCheckpoint(
        ckpt_path, monitor="val_fine_acc", mode="max",
        save_best_only=True, save_weights_only=True, verbose=1
    ),
    callbacks.CSVLogger(os.path.join(OUTPUT_DIR, "training_log.csv")),
]
 
# fase 1: warm-up — solo entrenan las cabezas, el tronco está congelado
# pongo cad_slow altísima para que mask_tronco siempre sea 0
print("\n=== FASE 1: warm-up ===")
base.get_layer("efficientnetb0").trainable = False
hmodel.cad_slow.assign(10**9)
hmodel.fit(train_ds, validation_data=val_ds,
           epochs=EPOCHS_WARMUP, callbacks=cbs, verbose=2)
 
# fase 2: fine-tuning — descongelamos las últimas 50 capas del backbone
# las primeras capas las dejo congeladas para no perder los detectores básicos
print("\n=== FASE 2: fine-tuning ===")
backbone_layer = base.get_layer("efficientnetb0")
for layer in backbone_layer.layers[:-50]:
    layer.trainable = False
 
# ahora sí activo la cadencia real del tronco
hmodel.cad_slow.assign(CAD_SLOW)
hmodel.global_step.assign(0)  # reseteo el contador para que las cadencias arranquen sincronizadas
 
hmodel.fit(train_ds, validation_data=val_ds,
           epochs=EPOCHS_FINETUNE, callbacks=cbs, verbose=2)
 
 

# EVALUACIÓN EN TEST
 
print("\n=== EVALUACION EN TEST ongoing ===")
 
# cargo los mejores pesos del checkpoint
hmodel.load_weights(ckpt_path)
 
# predigo sobre el conjunto de test
preds = base.predict(test_ds, verbose=0)
yt_f  = test_df["fine"].values
yt_c  = test_df["coarse"].values
yp_f  = np.argmax(preds["fine_out"],   axis=1)
yp_c  = np.argmax(preds["coarse_out"], axis=1)
 
print("\n--- Nivel grueso (benigno/maligno) ---")
print(classification_report(yt_c, yp_c, target_names=COARSE_CLASSES, digits=4))
 
print("\n--- Nivel fino (7 clases) ---")
print(classification_report(yt_f, yp_f, target_names=FINE_CLASSES, digits=4))
 
 
# MÉTRICAS FINALES Y MATRICES DE CONFUSIÓN
 
def tasa_consistencia(yp_c, yp_f):
    """
    Calcula qué porcentaje de predicciones son jerárquicamente consistentes,
    es decir, la clase fina predicha cae dentro de la superclase predicha.
    Complementa el término gamma de la pérdida — lo quiero cercano a 1.
    """
    superclase_de_fina = np.array([FINE_TO_COARSE[FINE_CLASSES[f]] for f in yp_f])
    return float(np.mean(superclase_de_fina == yp_c))
 
 
resultados = {
    "coarse_acc":       float(np.mean(yp_c == yt_c)),
    "coarse_balanced":  balanced_accuracy_score(yt_c, yp_c),
    "coarse_macro_f1":  f1_score(yt_c, yp_c, average="macro"),
    "fine_acc":         float(np.mean(yp_f == yt_f)),
    "fine_balanced":    balanced_accuracy_score(yt_f, yp_f),
    "fine_macro_f1":    f1_score(yt_f, yp_f, average="macro"),
    "consistency_rate": tasa_consistencia(yp_c, yp_f),
}
 
print("\nResumen métricas:")
for k, v in resultados.items():
    print(f"  {k:<22s}: {v:.4f}")
 
pd.DataFrame([resultados]).to_csv(
    os.path.join(OUTPUT_DIR, "final_metrics.csv"), index=False)
 
# matrices de confusión para los dos niveles
for nombre, y_real, y_pred, clases in [
    ("coarse", yt_c, yp_c, COARSE_CLASSES),
    ("fine",   yt_f, yp_f, FINE_CLASSES),
]:
    cm = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=clases, yticklabels=clases)
    plt.title(f"Matriz de confusión — {nombre}")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_{nombre}.png"), dpi=150)
    plt.close()
 
# se guardan los reports en un txt para citarlos en la memoria
ruta_report = os.path.join(OUTPUT_DIR, "classification_report.txt")
with open(ruta_report, "w") as f:
    f.write("===== NIVEL GRUESO (Benigno / Maligno) =====\n")
    f.write(classification_report(yt_c, yp_c, target_names=COARSE_CLASSES, digits=4))
    f.write("\n===== NIVEL FINO (7 clases) =====\n")
    f.write(classification_report(yt_f, yp_f, target_names=FINE_CLASSES, digits=4))
 
print(f"\n>> Salidas guardadas en {OUTPUT_DIR}/")