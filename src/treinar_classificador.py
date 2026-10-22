import json
from pathlib import Path

import tensorflow as tf


RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data/dataset"
SAIDA = RAIZ / "runs/mobilenet_v1"

TAMANHO = 224
LOTE = 16
EPOCAS = 20
SEMENTE = 42
TAXA_APRENDIZADO = 0.001


def main():
    if SAIDA.exists():
        raise FileExistsError(
            f"{SAIDA} já existe. Use outro nome para um novo experimento."
        )

    for parte in ("train", "val"):
        if not (DADOS / parte).is_dir():
            raise FileNotFoundError(f"Conjunto ausente: {DADOS / parte}")

    # Configura a GPU antes de criar modelos ou carregar os lotes.
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("TensorFlow não encontrou a GPU.")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.keras.utils.set_random_seed(SEMENTE)
    print("GPUs disponíveis:", gpus)

    treino = tf.keras.utils.image_dataset_from_directory(
        DADOS / "train",
        image_size=(TAMANHO, TAMANHO),
        batch_size=LOTE,
        label_mode="int",
        color_mode="rgb",
        shuffle=True,
        seed=SEMENTE,
    )

    classes = treino.class_names

    validacao = tf.keras.utils.image_dataset_from_directory(
        DADOS / "val",
        class_names=classes,
        image_size=(TAMANHO, TAMANHO),
        batch_size=LOTE,
        label_mode="int",
        color_mode="rgb",
        shuffle=False,
    )

    # Calcula os pesos usando somente as imagens de treinamento.
    contagens = [
        len(list((DADOS / "train" / nome).glob("*.png")))
        for nome in classes
    ]

    if any(quantidade == 0 for quantidade in contagens):
        raise RuntimeError("Existe uma classe sem imagens de treinamento.")

    total = sum(contagens)
    pesos = {
        indice: total / (len(classes) * quantidade)
        for indice, quantidade in enumerate(contagens)
    }

    for indice, nome in enumerate(classes):
        print(
            f"{indice}: {nome} — {contagens[indice]} imagens, "
            f"peso={pesos[indice]:.2f}"
        )

    # Prepara o próximo lote enquanto o atual é processado.
    treino = treino.prefetch(1)
    validacao = validacao.prefetch(1)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(TAMANHO, TAMANHO, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    aumento = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.03),
        tf.keras.layers.RandomZoom(0.10),
    ], name="aumento_dados")

    entrada = tf.keras.Input(
        shape=(TAMANHO, TAMANHO, 3), name="imagem_rgb"
    )

    x = aumento(entrada)
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5, offset=-1
    )(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    saida = tf.keras.layers.Dense(
        len(classes), activation="softmax", name="identidades"
    )(x)

    modelo = tf.keras.Model(entrada, saida)

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=TAXA_APRENDIZADO
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )

    SAIDA.mkdir(parents=True)

    configuracao = {
        "classes": classes,
        "contagens_treino": contagens,
        "pesos_classes": pesos,
        "tamanho": TAMANHO,
        "lote": LOTE,
        "epocas_maximas": EPOCAS,
        "semente": SEMENTE,
        "taxa_aprendizado": TAXA_APRENDIZADO,
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "base": "MobileNetV2",
        "pesos_base": "imagenet",
        "base_congelada": True,
        "entrada": "RGB com valores entre 0 e 255",
    }

    (SAIDA / "configuracao.json").write_text(
        json.dumps(configuracao, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(SAIDA / "melhor_modelo.keras"),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(str(SAIDA / "historico.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    modelo.summary()

    historico = modelo.fit(
        treino,
        validation_data=validacao,
        epochs=EPOCAS,
        class_weight=pesos,
        callbacks=callbacks,
    )

    melhor = min(
        range(len(historico.history["val_loss"])),
        key=lambda i: historico.history["val_loss"][i],
    )

    print(f"\nMelhor época por val_loss: {melhor + 1}")
    print(
        "Acurácia de validação nessa época:",
        f"{historico.history['val_accuracy'][melhor]:.2%}",
    )
    print("Arquivos do experimento:", SAIDA)


if __name__ == "__main__":
    main()