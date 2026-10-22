import csv
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
from sklearn.model_selection import train_test_split


RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "data/faces_candidatas"
SAIDA = RAIZ / "data/dataset"
RELATORIOS = RAIZ / "reports"
SEMENTE = 42


def salvar_csv(caminho, cabecalho, linhas):
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(cabecalho)
        escritor.writerows(linhas)


def main():
    if not ENTRADA.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {ENTRADA}")

    if SAIDA.exists():
        raise FileExistsError(
            f"{SAIDA} já existe. Renomeie-a antes de gerar outra divisão."
        )

    vistos = {}
    pessoas = defaultdict(list)
    duplicatas = []

    # Primeiro verifica todos os recortes, antes de criar os conjuntos.
    for pasta in sorted(ENTRADA.iterdir()):
        if not pasta.is_dir():
            continue

        arquivos = sorted(pasta.glob("*.png"))
        if not arquivos:
            raise RuntimeError(f"Pasta de identidade vazia: {pasta}")

        for arquivo in arquivos:
            imagem = cv2.imread(str(arquivo))

            if imagem is None or imagem.shape != (224, 224, 3):
                raise RuntimeError(f"Recorte inválido: {arquivo}")

            assinatura = hashlib.sha256(imagem.tobytes()).hexdigest()

            if assinatura in vistos:
                anterior = vistos[assinatura]

                if anterior.parent.name != pasta.name:
                    raise RuntimeError(
                        "Imagem idêntica com identidades diferentes: "
                        f"{anterior} e {arquivo}"
                    )

                duplicatas.append([
                    str(arquivo.relative_to(RAIZ)),
                    str(anterior.relative_to(RAIZ)),
                ])
                continue

            vistos[assinatura] = arquivo
            pessoas[pasta.name].append((arquivo, assinatura))

    if not pessoas:
        raise RuntimeError("Nenhuma imagem encontrada.")

    divisoes = {}

    for nome, amostras in sorted(pessoas.items()):
        if len(amostras) < 10:
            raise RuntimeError(f"Poucas imagens únicas para dividir: {nome}")

        treino, restante = train_test_split(
            amostras,
            test_size=0.30,
            random_state=SEMENTE,
        )

        validacao, teste = train_test_split(
            restante,
            test_size=0.50,
            random_state=SEMENTE,
        )

        divisoes[nome] = {
            "train": treino,
            "val": validacao,
            "test": teste,
        }

    # Confirma que nenhum conteúdo idêntico atravessou os conjuntos.
    hashes = {parte: set() for parte in ("train", "val", "test")}

    for partes in divisoes.values():
        for parte, amostras in partes.items():
            hashes[parte].update(assinatura for _, assinatura in amostras)

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if hashes[a] & hashes[b]:
            raise RuntimeError(f"Duplicata entre {a} e {b}.")

    registros = []

    for nome, partes in divisoes.items():
        for parte, amostras in partes.items():
            pasta_destino = SAIDA / parte / nome
            pasta_destino.mkdir(parents=True, exist_ok=True)

            for origem, assinatura in amostras:
                destino = pasta_destino / origem.name
                shutil.copy2(origem, destino)

                registros.append([
                    nome,
                    parte,
                    str(origem.relative_to(RAIZ)),
                    str(destino.relative_to(RAIZ)),
                    assinatura,
                ])

        print(
            f"{nome}: "
            f"treino={len(partes['train'])}, "
            f"validação={len(partes['val'])}, "
            f"teste={len(partes['test'])}"
        )

    RELATORIOS.mkdir(parents=True, exist_ok=True)

    salvar_csv(
        RELATORIOS / "divisao_dataset.csv",
        ["identidade", "conjunto", "origem", "destino", "sha256_pixels"],
        registros,
    )
    salvar_csv(
        RELATORIOS / "duplicatas_exatas.csv",
        ["duplicata_nao_copiada", "imagem_mantida"],
        duplicatas,
    )

    print(f"\nDuplicatas exatas não copiadas: {len(duplicatas)}")
    print(f"Total distribuído: {len(registros)}")
    print("Verificação: sem duplicatas exatas entre os conjuntos.")


if __name__ == "__main__":
    main()