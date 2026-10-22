from pathlib import Path
from collections import Counter

import cv2


RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data/downloads/lfw_home/lfw"
MODELO = RAIZ / "models/face_detection_yunet_2023mar.onnx"


def main():
    if not DADOS.is_dir():
        raise FileNotFoundError(f"Dataset não encontrado: {DADOS}")

    if not MODELO.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {MODELO}")

    detector = cv2.FaceDetectorYN.create(
        str(MODELO), "", (250, 250), 0.8, 0.3, 5000
    )

    total = Counter()

    print(
        f"{'Identidade':22} {'Total':>6} {'Uma':>6} "
        f"{'Zero':>6} {'Várias':>7} {'Erro':>6}",
        flush=True,
    )

    for pasta in sorted(DADOS.iterdir()):
        if not pasta.is_dir():
            continue

        arquivos = sorted(pasta.glob("*.jpg"))

        if len(arquivos) < 70:
            continue

        contagem = Counter()

        for arquivo in arquivos:
            imagem = cv2.imread(str(arquivo))

            if imagem is None:
                contagem["erro"] += 1
                continue

            altura, largura = imagem.shape[:2]
            detector.setInputSize((largura, altura))
            _, faces = detector.detect(imagem)

            quantidade = 0 if faces is None else len(faces)

            if quantidade == 0:
                contagem["zero"] += 1
            elif quantidade == 1:
                contagem["uma"] += 1
            else:
                contagem["varias"] += 1

        total.update(contagem)

        print(
            f"{pasta.name:22} {len(arquivos):6} "
            f"{contagem['uma']:6} {contagem['zero']:6} "
            f"{contagem['varias']:7} {contagem['erro']:6}",
            flush=True,
        )

    print(f"\nCandidatas ao treinamento: {total['uma']}")
    print(
        "Separadas para revisão:",
        total["zero"] + total["varias"] + total["erro"],
    )


if __name__ == "__main__":
    main()