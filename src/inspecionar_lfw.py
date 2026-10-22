from pathlib import Path

import cv2
import numpy as np


RAIZ = Path(__file__).resolve().parent.parent

DADOS = RAIZ / "data" / "downloads" / "lfw_home" / "lfw"
MODELO = RAIZ / "models" / "face_detection_yunet_2023mar.onnx"
SAIDA = RAIZ / "reports" / "inspecao_lfw.jpg"

MINIMO_IMAGENS = 70
AMOSTRAS_POR_PESSOA = 4
SEMENTE = 42


def main():
    if not DADOS.is_dir():
        raise FileNotFoundError(f"Dataset não encontrado em: {DADOS}")

    if not MODELO.is_file():
        raise FileNotFoundError(f"Modelo não encontrado em: {MODELO}")

    # Cada pasta representa uma identidade.
    pessoas = {}

    for pasta in sorted(DADOS.iterdir()):
        if pasta.is_dir():
            imagens = sorted(pasta.glob("*.jpg"))

            if len(imagens) >= MINIMO_IMAGENS:
                pessoas[pasta.name] = imagens

    if not pessoas:
        raise RuntimeError("Nenhuma identidade atende ao mínimo de imagens.")

    print(f"Identidades selecionadas: {len(pessoas)}")
    print(f"Imagens disponíveis: {sum(len(v) for v in pessoas.values())}")

    detector = cv2.FaceDetectorYN.create(
        str(MODELO), "", (250, 250), 0.8, 0.3, 5000
    )

    sorteio = np.random.default_rng(SEMENTE)
    linhas = []
    casos_para_revisar = 0

    for nome, arquivos in pessoas.items():
        print(f"{nome}: {len(arquivos)} imagens")
        indices = sorteio.choice(
            len(arquivos),
            size=AMOSTRAS_POR_PESSOA,
            replace=False,
        )

        miniaturas = []

        for indice in indices:
            arquivo = arquivos[int(indice)]
            imagem = cv2.imread(str(arquivo))

            if imagem is None:
                raise RuntimeError(f"Falha ao ler: {arquivo}")

            altura, largura = imagem.shape[:2]
            detector.setInputSize((largura, altura))
            _, faces = detector.detect(imagem)

            quantidade = 0 if faces is None else len(faces)

            if quantidade != 1:
                casos_para_revisar += 1
                print(f"  REVISAR: {arquivo.name} — {quantidade} faces")

            if faces is not None:
                for face in faces:
                    x, y, w, h = face[:4].astype(int)

                    cv2.rectangle(
                        imagem,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2,
                    )

            # Redimensionamos somente a visualização, após a detecção.
            miniatura = cv2.resize(imagem, (180, 180))

            cabecalho = np.full((45, 180, 3), 245, dtype=np.uint8)

            cv2.putText(
                cabecalho, nome.replace("_", " "), (5, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
            )
            cv2.putText(
                cabecalho, f"Faces detectadas: {quantidade}", (5, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
            )

            miniaturas.append(np.vstack([cabecalho, miniatura]))

        linhas.append(np.hstack(miniaturas))

    painel = np.vstack(linhas)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(SAIDA), painel):
        raise RuntimeError(f"Não foi possível salvar: {SAIDA}")

    print(f"\nAmostras com quantidade de faces diferente de 1: {casos_para_revisar}")
    print(f"Painel salvo em: {SAIDA}")


if __name__ == "__main__":
    main()