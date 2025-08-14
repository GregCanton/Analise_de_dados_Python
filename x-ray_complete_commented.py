#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detecção de Pneumonia em Raios‑X com HOG + LinearSVC (Calibrado)

Baseline explicável e reprodutível para classificação binária de radiografias
(NORMAL vs. PNEUMONIA) utilizando:

- HOG (Histogram of Oriented Gradients);
- LinearSVC com class_weight="balanced";
- Calibração de probabilidades (Platt/sigmoid) via CalibratedClassifierCV;
- Particionamento estratificado reprodutível (hold‑out);
- Relatório de classificação, matriz de confusão e curva ROC/AUC;
- CLI com três modos: train, predict, auto.

🆕 Ajuste: o CSV de predições **sempre** é salvo dentro de `report_dir` para evitar
`PermissionError` no Windows (por arquivo aberto/sem permissão).

Uso (exemplos)
-------------
Treinar:
    python x-ray_complete_commented_FIXED.py train \
        --data_dir "caminho/para/chestxrays.zip" \
        --model_out "models/pneumonia_linearSVC.pkl" \
        --report_dir "reports" \
        --eval_split 0.2

Predizer (lote):
    python x-ray_complete_commented_FIXED.py predict \
        --images_dir "caminho/para/chestxrays/test" \
        --model_path "models/pneumonia_linearSVC.pkl" \
        --out_csv "predicoes.csv" \
        --report_dir "reports"

Modo automático (treina se faltar modelo e depois prediz):
    python x-ray_complete_commented_FIXED.py auto \
        --data_dir "caminho/para/chestxrays.zip" \
        --images_dir "caminho/para/chestxrays/test" \
        --model_out "models/pneumonia_linearSVC.pkl" \
        --report_dir "reports" \
        --out_csv "predicoes.csv" \
        --eval_split 0.2

Notas importantes
-----------------
- Sem geração de dados sintéticos: o script consome imagens reais do seu dataset.
- Determinismo: HOG fixo; random_state=42.
- LGPD/Ética: uso educacional/pesquisa; não usar clinicamente sem validação regulatória.

Dependências mínimas
--------------------
python -m pip install opencv-python tqdm scikit-learn joblib matplotlib numpy
"""

from __future__ import annotations

# ============================ Imports básicos ============================ #
import argparse
import csv
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# ============================ Bibliotecas externas ======================= #
try:
    import cv2  # OpenCV
except Exception as exc:
    print(
        "Erro: OpenCV (cv2) não está instalado. Instale com: python -m pip install opencv-python",
        file=sys.stderr,
    )
    raise

try:
    from tqdm import tqdm  # barra de progresso
except Exception:
    # Fallback simples caso tqdm não esteja disponível
    def tqdm(x: Iterable, **_: dict) -> Iterable:  # type: ignore
        return x

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

import matplotlib
matplotlib.use("Agg")  # backend sem GUI para ambientes headless
import matplotlib.pyplot as plt

# ============================ Caminhos padrão ============================ #
# Ajuste estes caminhos para o seu ambiente, se desejar.
DATASET_DEFAULT_ZIP = r"C:\\Users\\gpcan\\OneDrive\\Área de Trabalho\\workspace\\data\\chestxrays.zip"
TEST_DEFAULT_DIR    = r"C:\\Users\\gpcan\\OneDrive\\Área de Trabalho\\workspace\\data\\chestxrays\\test"
MODEL_DEFAULT       = r"models\\pneumonia_linearSVC.pkl"
REPORTS_DEFAULT     = r"reports"

# ============================ Parâmetros do HOG ========================== #
# Mantidos como constantes para reprodutibilidade e clareza.
HOG_WIN_SIZE: Tuple[int, int]     = (128, 128)
HOG_BLOCK_SIZE: Tuple[int, int]   = (32, 32)
HOG_BLOCK_STRIDE: Tuple[int, int] = (16, 16)
HOG_CELL_SIZE: Tuple[int, int]    = (16, 16)
HOG_NBINS: int                    = 9

# ============================ Utilitários de ZIP/Path ==================== #

def _normalize_extraction_root(dest_dir: Path) -> Path:
    """Normaliza a raiz de extração do .zip, removendo duplicações de pasta.

    Exemplo corrigido: dest/chestxrays/chestxrays/train -> dest/train
    """
    try:
        subs = [d for d in dest_dir.iterdir() if d.is_dir()]
    except FileNotFoundError:
        return dest_dir

    if len(subs) == 1:
        inner = subs[0]
        if (inner / "train").exists() or (inner / "test").exists():
            for item in inner.iterdir():
                target = dest_dir / item.name
                if not target.exists():
                    try:
                        item.replace(target)
                    except Exception:
                        pass
            try:
                inner.rmdir()
            except Exception:
                pass
    return dest_dir


def extract_zip(zip_path: str, dest_dir: Optional[str] = None) -> str:
    """Extrai um arquivo .zip para dest_dir (ou pasta com mesmo nome) e normaliza estrutura."""
    zpath = Path(zip_path)
    if dest_dir is None:
        dest = zpath.with_suffix("")  # remove .zip
    else:
        dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(dest)
    dest = _normalize_extraction_root(dest)
    return str(dest)


def resolve_path_maybe_zip(input_path: str, prefer_subdir: Optional[str] = None) -> str:
    """Resolve um caminho que pode ser um .zip (extraindo) ou uma pasta já existente."""
    p = Path(input_path)
    if str(p).lower().endswith(".zip"):
        root = Path(extract_zip(str(p)))
    else:
        root = p

    if prefer_subdir:
        candidate = root / prefer_subdir
        if candidate.exists() and candidate.is_dir():
            return str(candidate)
    return str(root)


def list_images(root: str | Path) -> List[str]:
    """Lista de forma recursiva os caminhos de imagens suportadas em root."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    rootp = Path(root)
    return [str(p) for p in rootp.rglob("*") if p.suffix.lower() in exts]

# ============================ HOG e Carregamento ========================= #

def compute_hog(img_gray: np.ndarray) -> np.ndarray:
    """Computa descritores HOG para uma imagem em tons de cinza."""
    hog = cv2.HOGDescriptor(
        _winSize=HOG_WIN_SIZE,
        _blockSize=HOG_BLOCK_SIZE,
        _blockStride=HOG_BLOCK_STRIDE,
        _cellSize=HOG_CELL_SIZE,
        _nbins=HOG_NBINS,
    )
    feat = hog.compute(img_gray)
    return feat.reshape(-1)


def imread_grayscale_resized(path: str, size: Tuple[int, int] = HOG_WIN_SIZE) -> np.ndarray:
    """Lê uma imagem do disco em escala de cinza e a redimensiona para *size*.

    Implementa leitura robusta para Windows via np.fromfile+cv2.imdecode.
    Lança ValueError se a imagem não puder ser lida.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Falha ao ler imagem: {path}")
    if (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def _count_by_class(root_dir: str | Path) -> Dict[str, int]:
    """Conta o número de imagens por subpasta de classe (diagnóstico rápido)."""
    counts: Dict[str, int] = {}
    root = Path(root_dir)
    for d in sorted([x for x in root.iterdir() if x.is_dir()]):
        counts[d.name] = len(list_images(d))
    return counts


def load_dataset_from_class_folders(root_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Carrega o dataset a partir de uma estrutura de pastas por classe.

    Espera estrutura:
        root_dir/
          NORMAL/
          PNEUMONIA/
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Caminho não existe: {root_dir}")

    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"Nenhuma subpasta de classe encontrada em: {root_dir}")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    X: List[np.ndarray] = []
    y: List[int] = []
    paths: List[str] = []

    for cls in classes:
        cdir = root / cls
        imgs = list_images(cdir)
        if not imgs:
            print(f"[AVISO] nenhuma imagem em {cdir}", file=sys.stderr)
        for pth in tqdm(imgs, desc=f"HOG {cls}"):
            try:
                g = imread_grayscale_resized(pth)
                feat = compute_hog(g)
            except Exception as e:
                print(f"[AVISO] pulando arquivo ilegível/corrompido: {pth} - {e}", file=sys.stderr)
                continue
            X.append(feat)
            y.append(class_to_idx[cls])
            paths.append(str(pth))

    if not X:
        raise RuntimeError(
            "Nenhuma imagem útil carregada. Se estiver em OneDrive, marque a pasta como 'Sempre manter neste dispositivo'."
        )
    X_arr = np.vstack(X).astype(np.float32)
    y_arr = np.array(y, dtype=np.int32)
    return X_arr, y_arr, classes, paths

# ============================ Treinamento & Avaliação ==================== #

def train(data_dir: str, model_out: str, report_dir: str, eval_split: float = 0.2) -> Tuple[List[str], Dict[str, int]]:
    """Treina o modelo HOG + LinearSVC (calibrado) e salva relatórios/artefatos."""
    # 1) Resolve dataset (zip/pasta) priorizando 'train'
    data_root = resolve_path_maybe_zip(data_dir, prefer_subdir="train")

    try:
        counts = _count_by_class(data_root)
        print(f"[INFO] Imagens por classe em '{data_root}': {counts}")
        X, y, class_names, _ = load_dataset_from_class_folders(data_root)
    except Exception as e:
        # Fallback: tenta a raiz extraída quando 'train' não existe
        print(f"[AVISO] Falha ao carregar de '{data_root}' ({e}). Tentando raiz do dataset...", file=sys.stderr)
        data_root = resolve_path_maybe_zip(data_dir, prefer_subdir=None)
        counts = _count_by_class(data_root)
        print(f"[INFO] Imagens por classe em '{data_root}': {counts}")
        X, y, class_names, _ = load_dataset_from_class_folders(data_root)

    # 2) Split estratificado reprodutível
    sss = StratifiedShuffleSplit(n_splits=1, test_size=eval_split, random_state=42)
    (tr_idx, te_idx), = sss.split(X, y)
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    # 3) Pipeline: padronização (sem média) + SVM linear calibrado
    base = LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=5000)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    pipe = make_pipeline(StandardScaler(with_mean=False), clf)

    print("[INFO] Treinando modelo...")
    pipe.fit(Xtr, ytr)

    print("[INFO] Avaliando...")
    ypred = pipe.predict(Xte)
    try:
        yproba = pipe.predict_proba(Xte)
    except Exception:
        yproba = None

    # 4) Relatório de classificação
    rep = classification_report(yte, ypred, target_names=class_names, digits=4)
    cm = confusion_matrix(yte, ypred)

    # 5) Escrita de relatórios e figuras
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "classification_report.txt").write_text(rep, encoding="utf-8")

    # Matriz de confusão (com anotações)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Curva ROC (somente binário com probabilidades)
    if yproba is not None and len(class_names) == 2:
        pos_proba = yproba[:, 1]
        try:
            auc = roc_auc_score(yte, pos_proba)
        except Exception:
            auc = None
        fpr, tpr, _ = roc_curve(yte, pos_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=(f"ROC AUC={auc:.3f}" if auc is not None else "ROC"))
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / "roc_curve.png", dpi=150)
        plt.close()

    # 6) Serialização do modelo + metadados (para auditoria/reprodutibilidade)
    model_meta = {
        "class_names": class_names,
        "hog_params": {
            "win_size": HOG_WIN_SIZE,
            "block_size": HOG_BLOCK_SIZE,
            "block_stride": HOG_BLOCK_STRIDE,
            "cell_size": HOG_CELL_SIZE,
            "nbins": HOG_NBINS,
        },
    }
    Path(Path(model_out).parent).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipe, "meta": model_meta}, model_out)

    print(f"[OK] Modelo salvo em: {model_out}")
    print(f"[OK] Relatórios salvos em: {report_dir}")
    return model_meta["class_names"], counts

# ============================ Predição em lote =========================== #

def _softmax_from_decision(dec: np.ndarray) -> np.ndarray:
    """Converte decision_function em distribuição pseudo‑probabilística."""
    dec = np.atleast_1d(dec)
    if dec.ndim == 1:
        # Caso binário: logística simples
        p1 = 1.0 / (1.0 + np.exp(-dec[0]))
        return np.array([1 - p1, p1], dtype=float)
    # Multiclasse: softmax
    e = np.exp(dec - dec.max())
    return e / e.sum()


def predict(images_dir: str, model_path: str, out_csv: str, report_dir: str = REPORTS_DEFAULT) -> None:
    """Aplica o modelo salvo a todas as imagens em images_dir e escreve um CSV no diretório de relatórios.

    O arquivo de saída é sempre gravado dentro de `report_dir`, independentemente do caminho
    informado em `out_csv`. Apenas o nome-base (`Path(out_csv).name`) é usado para o arquivo final.
    """
    images_root = resolve_path_maybe_zip(images_dir, prefer_subdir="test")
    if not Path(images_root).exists():
        images_root = resolve_path_maybe_zip(images_dir)

    bundle = joblib.load(model_path)
    pipe = bundle["model"]
    class_names = bundle["meta"]["class_names"]

    files = list_images(images_root)
    if not files:
        raise ValueError(f"Nenhuma imagem encontrada em: {images_root}")

    records: List[Tuple] = []
    for fp in tqdm(files, desc="Predizendo"):
        try:
            g = imread_grayscale_resized(fp)
            feat = compute_hog(g).reshape(1, -1)
            pred = pipe.predict(feat)[0]
            try:
                proba = pipe.predict_proba(feat)[0]
            except Exception:
                df = pipe.decision_function(feat)
                proba = _softmax_from_decision(df)
            records.append((fp, class_names[int(pred)], *proba))
        except Exception as e:
            print(f"[AVISO] pulando arquivo ilegível/corrompido: {fp} - {e}", file=sys.stderr)
            continue

    # Força saída dentro de report_dir
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(out_csv).name

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["image_path", "pred_label"] + [f"proba_{c}" for c in class_names]
        writer.writerow(header)
        for r in records:
            writer.writerow(r)
    print(f"[OK] Predições salvas em: {out_path}")

# ============================ CLI ====================================== #

def run_auto(args: argparse.Namespace) -> None:
    """Executa: treina (se necessário) e depois prediz."""
    if not Path(args.model_out).exists():
        print(f"[INFO] Modelo '{args.model_out}' não existe. Iniciando treinamento automático...")
        train(args.data_dir, args.model_out, args.report_dir, args.eval_split)
    else:
        print(f"[INFO] Modelo encontrado: {args.model_out}")

    images_dir = args.images_dir
    if not Path(images_dir).exists():
        maybe_test = resolve_path_maybe_zip(args.data_dir, prefer_subdir="test")
        images_dir = maybe_test if Path(maybe_test).exists() else resolve_path_maybe_zip(args.data_dir, prefer_subdir=None)

    predict(images_dir, args.model_out, args.out_csv, args.report_dir)


def main(argv: Optional[List[str]] = None) -> None:
    """Ponto de entrada CLI."""
    parser = argparse.ArgumentParser(description="Pneumonia - HOG + LinearSVC")
    sub = parser.add_subparsers(dest="cmd")

    # Treino
    p_train = sub.add_parser("train", help="Treinar o modelo")
    p_train.add_argument("--data_dir", default=DATASET_DEFAULT_ZIP, help="Pasta OU .zip (raiz ou 'train')")
    p_train.add_argument("--model_out", default=MODEL_DEFAULT)
    p_train.add_argument("--report_dir", default=REPORTS_DEFAULT)
    p_train.add_argument("--eval_split", type=float, default=0.2)

    # Predição
    p_pred = sub.add_parser("predict", help="Predizer em imagens")
    p_pred.add_argument("--images_dir", default=TEST_DEFAULT_DIR, help="Pasta OU .zip (raiz ou 'test')")
    p_pred.add_argument("--model_path", default=MODEL_DEFAULT)
    p_pred.add_argument("--out_csv", default="predicoes.csv")
    p_pred.add_argument("--report_dir", default=REPORTS_DEFAULT)

    # Automático
    p_auto = sub.add_parser("auto", help="Treinar (se necessário) e predizer (modo one‑shot)")
    p_auto.add_argument("--data_dir", default=DATASET_DEFAULT_ZIP)
    p_auto.add_argument("--images_dir", default=TEST_DEFAULT_DIR)
    p_auto.add_argument("--model_out", default=MODEL_DEFAULT)
    p_auto.add_argument("--report_dir", default=REPORTS_DEFAULT)
    p_auto.add_argument("--out_csv", default="predicoes.csv")
    p_auto.add_argument("--eval_split", type=float, default=0.2)

    args = parser.parse_args(argv)

    # Sem argumentos → modo AUTO com defaults
    if args.cmd is None:
        print("[INFO] Nenhum subcomando informado. Rodando em modo AUTO com caminhos padrão.")
        auto_args = argparse.Namespace(
            data_dir=DATASET_DEFAULT_ZIP,
            images_dir=TEST_DEFAULT_DIR,
            model_out=MODEL_DEFAULT,
            report_dir=REPORTS_DEFAULT,
            out_csv="predicoes.csv",
            eval_split=0.2,
        )
        return run_auto(auto_args)

    if args.cmd == "train":
        train(args.data_dir, args.model_out, args.report_dir, args.eval_split)
    elif args.cmd == "predict":
        if not Path(args.model_path).exists():
            print(f"[AVISO] Modelo '{args.model_path}' não encontrado. Treinando automaticamente antes de predizer...")
            train(DATASET_DEFAULT_ZIP, args.model_path, REPORTS_DEFAULT, 0.2)
        predict(args.images_dir, args.model_path, args.out_csv, args.report_dir)
    elif args.cmd == "auto":
        run_auto(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
