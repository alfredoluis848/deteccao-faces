import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from preparar_faces import recortar_face, TAMANHO


RAIZ = Path(__file__).resolve().parent.parent
EXPERIMENTO = RAIZ / "runs/mobilenet_v1"
DETECTOR = RAIZ / "models/face_detection_yunet_2023mar.onnx"


def main():
    parser = argparse.ArgumentParser(
        description="Detecta faces e classifica entre as identidades cadastradas."
    )
    parser.add_argument("imagem", type=Path, help="Fotografia de entrada.")
    parser.add_argument(
        "--saida",
        type=Path,
        default=RAIZ / "reports/reconhecimento.jpg",
        help="Imagem de saída, com extensão .jpg ou .png.",
    )
    args = parser.parse_args()

    if args.saida.suffix.lower() not in (".jpg", ".png"):
        raise ValueError("A saída deve ter extensão .jpg ou .png.")

    saida_json = args.saida.with_suffix(".json")

    if args.imagem.resolve() in (
        args.saida.resolve(),
        saida_json.resolve(),
    ):
        raise ValueError("A saída não pode substituir o arquivo de entrada.")

    imagem = cv2.imread(str(args.imagem))
    if imagem is None:
        raise RuntimeError(f"Não foi possível ler: {args.imagem}")

    configuracao = json.loads(
        (EXPERIMENTO / "configuracao.json").read_text(encoding="utf-8")
    )
    classes = configuracao["classes"]

    if configuracao["tamanho"] != TAMANHO:
        raise ValueError("O tamanho do recorte difere do usado no treinamento.")

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Limita o tamanho de fotografias grandes, preservando a proporção.
    altura_original, largura_original = imagem.shape[:2]
    escala = min(1.0, 1280 / max(altura_original, largura_original))

    if escala < 1.0:
        imagem = cv2.resize(
            imagem,
            (
                max(1, round(largura_original * escala)),
                max(1, round(altura_original * escala)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    altura, largura = imagem.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        str(DETECTOR), "", (largura, altura), 0.8, 0.3, 5000
    )
    _, faces = detector.detect(imagem)

    recortes = []
    faces_validas = []

    if faces is not None:
        for face in faces:
            recorte = recortar_face(imagem, face)

            if recorte is not None:
                rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
                recortes.append(rgb)
                faces_validas.append(face)

    resultados = []

    if recortes:
        modelo = tf.keras.models.load_model(
            EXPERIMENTO / "melhor_modelo.keras",
            compile=False,
        )

        # Processa os rostos em lotes, limitando o uso de memória.
        pontuacoes = modelo.predict(
            np.stack(recortes),
            batch_size=configuracao["lote"],
            verbose=0,
        )

        for face, scores in zip(faces_validas, pontuacoes):
            indice = int(np.argmax(scores))
            nome = classes[indice]
            score = float(scores[indice])
            x, y, w, h = map(int, face[:4])

            resultados.append({
                "classe_prevista": nome,
                "score_classificacao": score,
                "score_deteccao": float(face[14]),
                "caixa_xywh": [x, y, w, h],
            })

            cv2.rectangle(
                imagem, (x, y), (x + w, y + h), (0, 255, 0), 2
            )
            cv2.putText(
                imagem,
                f"{nome.replace('_', ' ')}: {score:.2f}",
                (max(0, x), max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    args.saida.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(args.saida), imagem):
        raise RuntimeError("Não foi possível salvar a imagem de saída.")

    relatorio = {
        "entrada": str(args.imagem.resolve()),
        "tamanho_processado_wh": [largura, altura],
        "coordenadas": "relativas à imagem processada e salva",
        "faces_detectadas": 0 if faces is None else len(faces),
        "faces_classificadas": len(resultados),
        "resultados": resultados,
    }

    saida_json.write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for resultado in resultados:
        print(
            resultado["classe_prevista"],
            f"score={resultado['score_classificacao']:.3f}",
        )

    print(f"Faces classificadas: {len(resultados)}")
    print("Imagem salva em:", args.saida)
    print("Relatório salvo em:", saida_json)


if __name__ == "__main__":
    main()