import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Salva gráficos sem abrir janelas.

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


RAIZ = Path(__file__).resolve().parent.parent
EXPERIMENTO = RAIZ / "runs/mobilenet_v1"
TESTE = RAIZ / "data/dataset/test"
SAIDA = EXPERIMENTO / "avaliacao"


def gerar_curvas():
    with (EXPERIMENTO / "historico.csv").open(
        newline="", encoding="utf-8"
    ) as arquivo:
        historico = list(csv.DictReader(arquivo))

    epocas = [int(linha["epoch"]) + 1 for linha in historico]
    perdas_val = [float(linha["val_loss"]) for linha in historico]
    melhor_epoca = epocas[int(np.argmin(perdas_val))]

    figura, eixos = plt.subplots(1, 2, figsize=(12, 4))

    for eixo, metrica, titulo in zip(
        eixos,
        ("accuracy", "loss"),
        ("Acurácia", "Perda"),
    ):
        eixo.plot(
            epocas,
            [float(linha[metrica]) for linha in historico],
            label="Treinamento",
        )
        eixo.plot(
            epocas,
            [float(linha[f"val_{metrica}"]) for linha in historico],
            label="Validação",
        )
        eixo.axvline(
            melhor_epoca, color="gray", linestyle="--",
            label="Época selecionada",
        )
        eixo.set(xlabel="Época", ylabel=titulo, title=titulo)
        eixo.grid(alpha=0.3)
        eixo.legend()

    figura.tight_layout()
    figura.savefig(SAIDA / "curvas_treinamento.png", dpi=150)
    plt.close(figura)


def main():
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    configuracao = json.loads(
        (EXPERIMENTO / "configuracao.json").read_text(encoding="utf-8")
    )
    classes = configuracao["classes"]
    tamanho = configuracao["tamanho"]
    indices = list(range(len(classes)))

    teste = tf.keras.utils.image_dataset_from_directory(
        TESTE,
        class_names=classes,
        image_size=(tamanho, tamanho),
        batch_size=configuracao["lote"],
        label_mode="int",
        color_mode="rgb",
        shuffle=False,
    )

    modelo = tf.keras.models.load_model(
        EXPERIMENTO / "melhor_modelo.keras",
        compile=False,
    )

    reais = []
    previstas = []

    # Mantém cada previsão associada ao rótulo do mesmo lote.
    for imagens, rotulos in teste:
        pontuacoes = modelo(imagens, training=False).numpy()
        reais.extend(rotulos.numpy().tolist())
        previstas.extend(np.argmax(pontuacoes, axis=1).tolist())

    reais = np.asarray(reais)
    previstas = np.asarray(previstas)

    texto = classification_report(
        reais,
        previstas,
        labels=indices,
        target_names=classes,
        digits=3,
        zero_division=0,
    )

    relatorio = classification_report(
        reais,
        previstas,
        labels=indices,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    # Referência simples: sempre escolher a maior classe do treinamento.
    classe_majoritaria = int(np.argmax(configuracao["contagens_treino"]))
    previsao_basica = np.full_like(reais, classe_majoritaria)

    resumo = {
        "total_teste": int(len(reais)),
        "acuracia": float(accuracy_score(reais, previstas)),
        "f1_macro": float(relatorio["macro avg"]["f1-score"]),
        "acuracia_referencia": float(
            accuracy_score(reais, previsao_basica)
        ),
        "relatorio": relatorio,
    }

    SAIDA.mkdir(parents=True, exist_ok=True)

    (SAIDA / "relatorio.txt").write_text(texto, encoding="utf-8")
    (SAIDA / "metricas.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    matriz = confusion_matrix(reais, previstas, labels=indices)
    nomes = [nome.replace("_", " ") for nome in classes]

    figura, eixo = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=nomes,
    ).plot(
        ax=eixo,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45,
        colorbar=False,
    )

    eixo.set(
        xlabel="Identidade prevista",
        ylabel="Identidade real",
        title="Matriz de confusão — conjunto de teste",
    )
    figura.tight_layout()
    figura.savefig(SAIDA / "matriz_confusao.png", dpi=150)
    plt.close(figura)

    gerar_curvas()

    print("\n", texto)
    print(f"Acurácia no teste: {resumo['acuracia']:.2%}")
    print(f"F1 macro: {resumo['f1_macro']:.3f}")
    print(
        "Acurácia escolhendo sempre a maior classe:",
        f"{resumo['acuracia_referencia']:.2%}",
    )
    print("Resultados salvos em:", SAIDA)


if __name__ == "__main__":
    main()