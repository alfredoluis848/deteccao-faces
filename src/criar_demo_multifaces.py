import json
from pathlib import Path

import cv2
import numpy as np


RAIZ = Path(__file__).resolve().parent.parent

TESTE = RAIZ / "data/dataset/test"
ORIGINAIS = RAIZ / "data/downloads/lfw_home/lfw"
CONFIGURACAO = RAIZ / "runs/mobilenet_v1/configuracao.json"
SAIDA = RAIZ / "reports/demo_multifaces"

COLUNAS = 4
TAMANHO_FOTO = 250
ESPACO = 16
ALTURA_LEGENDA = 35
SEMENTE = 42


def main():
    configuracao = json.loads(
        CONFIGURACAO.read_text(encoding="utf-8")
    )
    classes = configuracao["classes"]

    # Calcula quantas linhas são necessárias para acomodar as pessoas.
    linhas = (len(classes) + COLUNAS - 1) // COLUNAS
    largura = ESPACO + COLUNAS * (TAMANHO_FOTO + ESPACO)
    altura = ESPACO + linhas * (
        TAMANHO_FOTO + ALTURA_LEGENDA + ESPACO
    )

    # Cria uma imagem vazia com fundo cinza-claro.
    painel = np.full((altura, largura, 3), 235, dtype=np.uint8)

    sorteio = np.random.default_rng(SEMENTE)
    referencias = []

    for indice, pessoa in enumerate(classes):
        candidatos = sorted((TESTE / pessoa).glob("*.png"))

        if not candidatos:
            raise FileNotFoundError(f"Sem imagens de teste para: {pessoa}")

        escolhido = candidatos[int(sorteio.integers(len(candidatos)))]

        # Recupera a fotografia original correspondente ao recorte de teste.
        original = ORIGINAIS / pessoa / f"{escolhido.stem}.jpg"
        imagem = cv2.imread(str(original))

        if imagem is None:
            raise RuntimeError(f"Não foi possível ler: {original}")

        if imagem.shape[:2] != (TAMANHO_FOTO, TAMANHO_FOTO):
            raise ValueError(f"Dimensões inesperadas no original: {original}")

        linha, coluna = divmod(indice, COLUNAS)

        x = ESPACO + coluna * (TAMANHO_FOTO + ESPACO)
        y = ESPACO + linha * (
            TAMANHO_FOTO + ALTURA_LEGENDA + ESPACO
        )

        painel[
            y:y + TAMANHO_FOTO,
            x:x + TAMANHO_FOTO,
        ] = imagem

        numero = indice + 1

        cv2.putText(
            painel,
            f"Foto {numero}",
            (x, y + TAMANHO_FOTO + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

        referencias.append({
            "foto": numero,
            "identidade_esperada": pessoa,
            "original": str(original.relative_to(RAIZ)),
            "regiao_da_foto_xywh": [
                x, y, TAMANHO_FOTO, TAMANHO_FOTO
            ],
        })

        print(f"Foto {numero}: {pessoa} — {original.name}")

    SAIDA.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(SAIDA / "entrada.png"), painel):
        raise RuntimeError("Não foi possível salvar a montagem.")

    (SAIDA / "referencia.json").write_text(
        json.dumps(referencias, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nMontagem salva em:", SAIDA / "entrada.png")
    print("Respostas esperadas salvas em:", SAIDA / "referencia.json")


if __name__ == "__main__":
    main()