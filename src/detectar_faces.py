from pathlib import Path

import cv2


# Resolve o modelo a partir da localização deste arquivo.
RAIZ = Path(__file__).resolve().parent.parent
MODELO = RAIZ / "models" / "face_detection_yunet_2023mar.onnx"

CAMERA = 0
LIMIAR_DETECCAO = 0.8


def main():
    if not MODELO.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {MODELO}")

    # Carrega a rede uma única vez, antes de processar o vídeo.
    detector = cv2.FaceDetectorYN.create(
        model=str(MODELO),
        config="",
        input_size=(640, 480),
        score_threshold=LIMIAR_DETECCAO,
        nms_threshold=0.3,
        top_k=5000,
        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
        target_id=cv2.dnn.DNN_TARGET_CPU,
    )

    camera = cv2.VideoCapture(CAMERA)

    try:
        if not camera.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera {CAMERA}.")

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Pressione Q na janela para encerrar.")

        while True:
            sucesso, quadro = camera.read()

            if not sucesso:
                raise RuntimeError("Não foi possível capturar um quadro.")

            # A câmera pode entregar uma resolução diferente da solicitada.
            altura, largura = quadro.shape[:2]
            detector.setInputSize((largura, altura))

            # Executa a inferência sobre o quadro original.
            _, faces = detector.detect(quadro)
            quantidade = 0 if faces is None else len(faces)

            if faces is not None:
                for face in faces:
                    x, y, w, h = face[:4].astype(int)
                    pontuacao = float(face[14])

                    cv2.rectangle(
                        quadro, (x, y), (x + w, y + h), (0, 255, 0), 2
                    )

                    cv2.putText(
                        quadro,
                        f"Face: {pontuacao:.2f}",
                        (max(0, x), max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            cv2.putText(
                quadro,
                f"Faces: {quantidade}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Deteccao facial - Q para sair", quadro)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()