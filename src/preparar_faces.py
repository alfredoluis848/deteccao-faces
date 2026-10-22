import csv
from collections import Counter
from pathlib import Path

import cv2


RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data/downloads/lfw_home/lfw"
MODELO = RAIZ / "models/face_detection_yunet_2023mar.onnx"
SAIDA = RAIZ / "data/faces_candidatas"
RELATORIO = RAIZ / "reports/preparacao_faces.csv"

TAMANHO = 224
MARGEM = 0.15


def recortar_face(imagem, face):
    altura, largura = imagem.shape[:2]
    x, y, w, h = map(float, face[:4])

    # Acrescenta margem e limita o recorte às bordas da imagem.
    esquerda = max(0, int(x - MARGEM * w))
    topo = max(0, int(y - MARGEM * h))
    direita = min(largura, int(x + w + MARGEM * w))
    base = min(altura, int(y + h + MARGEM * h))

    if direita <= esquerda or base <= topo:
        return None

    recorte = imagem[topo:base, esquerda:direita]

    # Completa o lado menor, mantendo as proporções do rosto.
    h_recorte, w_recorte = recorte.shape[:2]
    lado = max(h_recorte, w_recorte)

    acima = (lado - h_recorte) // 2
    abaixo = lado - h_recorte - acima
    esquerda_extra = (lado - w_recorte) // 2
    direita_extra = lado - w_recorte - esquerda_extra

    quadrado = cv2.copyMakeBorder(
        recorte,
        acima,
        abaixo,
        esquerda_extra,
        direita_extra,
        cv2.BORDER_REPLICATE,
    )

    interpolacao = (
        cv2.INTER_AREA if lado > TAMANHO else cv2.INTER_LINEAR
    )
    return cv2.resize(
        quadrado, (TAMANHO, TAMANHO), interpolation=interpolacao
    )


def main():
    if not DADOS.is_dir() or not MODELO.is_file():
        raise FileNotFoundError("Confira os caminhos do dataset e do modelo.")

    # Evita misturar resultados de execuções com configurações diferentes.
    if SAIDA.exists():
        raise FileExistsError(
            f"A pasta {SAIDA} já existe. Renomeie-a antes de executar novamente."
        )

    detector = cv2.FaceDetectorYN.create(
        str(MODELO), "", (250, 250), 0.8, 0.3, 5000
    )

    SAIDA.mkdir(parents=True)
    RELATORIO.parent.mkdir(parents=True, exist_ok=True)
    totais = Counter()

    with RELATORIO.open("w", newline="", encoding="utf-8") as arquivo_csv:
        registro = csv.writer(arquivo_csv)
        registro.writerow(["identidade", "origem", "status", "recorte"])

        for pasta in sorted(DADOS.iterdir()):
            if not pasta.is_dir():
                continue

            arquivos = sorted(pasta.glob("*.jpg"))
            if len(arquivos) < 70:
                continue

            salvas = 0

            for arquivo in arquivos:
                imagem = cv2.imread(str(arquivo))
                destino_relativo = ""
                status = "erro_leitura"

                if imagem is not None:
                    altura, largura = imagem.shape[:2]
                    detector.setInputSize((largura, altura))
                    _, faces = detector.detect(imagem)

                    quantidade = 0 if faces is None else len(faces)
                    status = "sem_face" if quantidade == 0 else "multiplas_faces"

                    if quantidade == 1:
                        recorte = recortar_face(imagem, faces[0])
                        status = "recorte_invalido"

                        if recorte is not None:
                            destino = SAIDA / pasta.name / f"{arquivo.stem}.png"
                            destino.parent.mkdir(parents=True, exist_ok=True)

                            if not cv2.imwrite(str(destino), recorte):
                                raise RuntimeError(f"Falha ao salvar: {destino}")

                            destino_relativo = str(destino.relative_to(RAIZ))
                            status = "salva"
                            salvas += 1

                totais[status] += 1
                registro.writerow([
                    pasta.name,
                    str(arquivo.relative_to(RAIZ)),
                    status,
                    destino_relativo,
                ])

            print(f"{pasta.name}: {salvas} recortes salvos", flush=True)

    print("\nResumo:", dict(totais))
    print("Recortes:", SAIDA)
    print("Relatório:", RELATORIO)


if __name__ == "__main__":
    main()