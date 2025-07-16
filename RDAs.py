import os
import json
import math
import heapq
import csv
import logging
import sys
from collections import Counter
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import clip

# ===== Logging setup with colored output =====
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

class ColoredFormatter(logging.Formatter):
    """
    Logging Formatter to add colors based on log level
    """
    FORMATS = {
        logging.INFO:    GREEN + "%(asctime)s - " + "%(levelname)s" + RESET + " - %(message)s",
        logging.WARNING: RED   + "%(asctime)s - " + "%(levelname)s" + RESET + " - %(message)s",
        logging.ERROR:   RED   + "%(asctime)s - " + "%(levelname)s" + RESET + " - %(message)s",
        logging.CRITICAL:RED   + "%(asctime)s - " + "%(levelname)s" + RESET + " - %(message)s",
        logging.DEBUG:   GREEN + "%(asctime)s - " + "%(levelname)s" + RESET + " - %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

# Configure root logger
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter())
logger = logging.getLogger("rdas_logger")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ===== Configuration loading =====
cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(cfg_path, "r") as f:
    config = json.load(f)

DEVICE = config["RDA"]["device"]
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)


def get_model_and_dataset(config):
    try:
        clip_model, _ = clip.load(config["RDA"]["model_name"], device=DEVICE)
        model = clip_model.visual.float().to(DEVICE)
        del clip_model
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        raise

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    data_root = config["RDA"]["dataset_path"]
    name = config["RDA"]["dataset_name"].upper()
    if name == "CIFAR10":
        train_dataset = datasets.CIFAR10(data_root, train=True,  download=True, transform=valid_transform)
        test_dataset  = datasets.CIFAR10(data_root, train=False, download=True, transform=valid_transform)
    elif name == "CIFAR100":
        train_dataset = datasets.CIFAR100(data_root, train=True,  download=True, transform=valid_transform)
        test_dataset  = datasets.CIFAR100(data_root, train=False, download=True, transform=valid_transform)
    else:
        logger.error(f"Unsupported dataset: {config['RDA']['dataset_name']}")
        raise ValueError(f"Unsupported dataset: {config['RDA']['dataset_name']}")

    train_loader = DataLoader(train_dataset,
                              batch_size=config["RDA"]["batch_size"],
                              shuffle=True,
                              num_workers=config["RDA"]["num_workers"])
    test_loader  = DataLoader(test_dataset,
                              batch_size=config["RDA"]["batch_size"],
                              shuffle=False,
                              num_workers=config["RDA"]["num_workers"])

    return model, (train_loader, test_loader)


def evaluate_model_classification(model, train_loader, eval_loader, device, num_classes, l=12):
    model.eval()
    model.to(device)
    features_list, labels_list = [], []

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.cpu()
            _, feats = model(inputs, l)
            features_list.append(feats.cpu())
            labels_list.append(labels)

    features_train = torch.cat(features_list, dim=0)
    labels_train   = torch.cat(labels_list,   dim=0)
    N, D = features_train.shape

    ones = torch.ones(N, 1)
    F_train = torch.cat([features_train, ones], dim=1)
    Y_train = torch.zeros(N, num_classes)
    Y_train[torch.arange(N), labels_train] = 1.0

    lstsq_res = torch.linalg.lstsq(F_train, Y_train)
    W_all = lstsq_res.solution
    W = W_all[:-1, :].t()
    b = W_all[-1, :].t()

    classifier = nn.Linear(D, num_classes).to(device)
    classifier.weight.data.copy_(W.to(device))
    classifier.bias.data.copy_(b.to(device))

    features_list, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs = inputs.to(device)
            labels = labels.cpu()
            _, feats = model(inputs, l)
            features_list.append(feats.cpu())
            labels_list.append(labels)

    features_eval = torch.cat(features_list, dim=0).to(DEVICE)
    labels_eval   = torch.cat(labels_list,   dim=0).to(DEVICE)

    logits = classifier(features_eval)
    preds  = logits.argmax(dim=1)
    accuracy = (preds == labels_eval).float().mean().item() * 100

    logger.info(f"Evaluation accuracy: {accuracy:.2f}%")
    return accuracy, classifier


def svid(m: torch.Tensor, r: int):
    single = False
    if m.dim() == 2:
        m = m.unsqueeze(0)
        single = True

    m_sign = m.sign()
    m_abs  = m.abs()

    u, s, vh = torch.linalg.svd(m_abs, full_matrices=False)
    u_r  = u[..., :r]
    s_r  = s[..., :r]
    vh_r = vh[..., :r, :]

    approx = torch.matmul(u_r * s_r.unsqueeze(-2), vh_r)
    error = m - m_sign * approx
    error_norm = error.norm(p='fro', dim=(-2, -1))

    if single:
        return u_r[0], s_r[0], vh_r[0], m_sign, error_norm[0].item()
    return u_r, s_r, vh_r, m_sign, error_norm


def reshape_feature(feature: torch.Tensor):
    batch_size, D = feature.shape
    root = int(math.isqrt(D))
    for n in range(root, 0, -1):
        if D % n == 0:
            m = D // n
            break
    d1, d2 = n, m
    if d1 * d2 <= 0:
        logger.error(f"Reshape error: invalid dimensions {d1}x{d2} for D={D}")
        raise AssertionError("Invalid reshape dimensions: last two dims must multiply to >0.")

    return feature.view(batch_size, n, m)


def flatten_feature(feature: torch.Tensor) -> torch.Tensor:
    batch_size, d1, d2 = feature.shape
    if d1 * d2 <= 0:
        logger.error("Flatten error: The last two dims must be >0.")
        raise AssertionError("Invalid flatten dims.")
    return feature.view(batch_size, d1 * d2)


class _HuffmanNode:
    __slots__ = ("freq", "symbol", "left", "right")
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq


def _build_huffman_tree(freq_dict):
    heap = [_HuffmanNode(f, s) for s, f in freq_dict.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = _HuffmanNode(n1.freq + n2.freq, None, n1, n2)
        heapq.heappush(heap, merged)
    return heap[0]


def _build_code_map(node, prefix="", code_map=None):
    if code_map is None:
        code_map = {}
    if node.symbol is not None:
        code_map[node.symbol] = prefix or "0"
    else:
        _build_code_map(node.left,  prefix + "0", code_map)
        _build_code_map(node.right, prefix + "1", code_map)
    return code_map


def compress_tensor(x: torch.Tensor, K: int, method: str="huffman"):
    if x.dim() != 3:
        logger.error(f"Compression error: Expected 3D tensor, got {x.dim()}D.")
        raise AssertionError("The input tensor must have 3 dimensions (B, m, n).")
    B, m, n = x.shape

    x_flat = x.reshape(-1)
    x_min = float(x_flat.min().item())
    x_max = float(x_flat.max().item())
    scale = ((2**K - 1) / (x_max - x_min)) if x_max > x_min else 1.0

    q = torch.clamp(torch.round((x - x_min) * scale), 0, 2**K - 1).to(torch.int32)
    q_list = q.reshape(-1).tolist()

    if method.lower() == "huffman":
        freq = Counter(q_list)
        tree = _build_huffman_tree(freq)
        code_map = _build_code_map(tree)
        bitstream = "".join(code_map[s] for s in q_list)
        metadata = {"Bmn": (B, m, n), "x_min": x_min, "x_max": x_max,
                    "K": K, "method": "huffman", "code_map": code_map}
        return bitstream, metadata

    logger.error(f"Unsupported compression method: {method}")
    raise ValueError(f"Unsupported compression method: {method}")


def decompress_tensor(bitstream: str, metadata: dict) -> torch.Tensor:
    B, m, n = metadata["Bmn"]
    x_min = metadata["x_min"]
    x_max = metadata["x_max"]
    K     = metadata["K"]
    method= metadata["method"].lower()

    scale = ((2**K - 1) / (x_max - x_min)) if x_max > x_min else 1.0
    if method == "huffman":
        code_map = metadata["code_map"]
        root = {}
        for sym, code in code_map.items():
            node = root
            for b in code:
                node = node.setdefault(b, {})
            node["sym"] = sym

        decoded = []
        node = root
        for b in bitstream:
            node = node[b]
            if "sym" in node:
                decoded.append(node["sym"])
                node = root

        q = torch.tensor(decoded, dtype=torch.int32).view(B, m, n)
    else:
        logger.error(f"Unsupported decoding method: {method}")
        raise ValueError(f"Unsupported decoding method: {method}")

    x_hat = q.to(torch.float32) / scale + x_min
    return x_hat


def save_results_to_csv(bit_stream_length, accuracy,
                        csv_file='results.tsv', delimiter='\t'):
    try:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(['quantify_bits','decomposition_rank','bitstream_length','task_performance'])
            for key, length in bit_stream_length.items():
                parts = key.split('_')
                rank = int(parts[1]); bits = int(parts[3])
                perf = accuracy[key]
                writer.writerow([bits, rank, length, f'{perf:.4f}'])
        logger.info(f"Experience configuration saved to {csv_file}")
    except Exception as e:
        logger.error(f"Failed to save results to CSV: {e}")
        raise


if __name__ == "__main__":
    model, (train_loader, test_loader) = get_model_and_dataset(config)
    model.eval()
    logger.info("Model loads successfully, evaluating and training classifier...")

    accuracy, classifier = evaluate_model_classification(
        model,
        train_loader,
        test_loader,
        device=DEVICE,
        num_classes=config["RDA"]["num_classes"],
        l=len(model.transformer.resblocks)-1
    )

    data_iterator = tqdm(test_loader, desc="Evaluating model", ncols=150, leave=False)
    bit_stream_length_list = {f"rank_{r}_bits_{b}": 0 for r in config["RDA"]["rank_list"] for b in config["RDA"]["quantization_bit_list"]}
    accuracy_list = bit_stream_length_list.copy()

    for images, labels in data_iterator:
        images = images.to(DEVICE)
        _, features = model(images, 11)
        for rank in config["RDA"]["rank_list"]:
            if rank != 0:
                reshaped = reshape_feature(features)
                f_u, f_s, f_vh, f_sign, err = svid(reshaped, rank)
            else:
                f_u = features; f_s = None; f_vh = None; f_sign = None

            for bits in config["RDA"]["quantization_bit_list"]:
                if rank != 0:
                    bu, mu = compress_tensor(f_u, bits)
                    bs, ms = compress_tensor(f_s.unsqueeze(-1), bits)
                    bv, mv = compress_tensor(f_vh, bits)
                    bsign, msign = compress_tensor(f_sign, 1)
                    length = len(bu) + len(bs) + len(bv) + len(bsign)
                    bit_stream_length_list[f"rank_{rank}_bits_{bits}"] += length

                    du = decompress_tensor(bu, mu).to(DEVICE)
                    ds = decompress_tensor(bs, ms).squeeze(-1).to(DEVICE)
                    dv = decompress_tensor(bv, mv).to(DEVICE)
                    dsig = decompress_tensor(bsign, msign).to(DEVICE)
                    scaled = du * ds.unsqueeze(-2)
                    recon = torch.matmul(scaled, dv)
                    features_hat = dsig * recon
                else:
                    bori, mori = compress_tensor(features.unsqueeze(-1), bits)
                    bit_stream_length_list[f"rank_{rank}_bits_{bits}"] += len(bori)
                    features_hat = decompress_tensor(bori, mori).to(DEVICE)

                flat = flatten_feature(features_hat)
                logits = classifier(flat)
                preds  = logits.argmax(dim=1)
                correct= (preds == labels.to(DEVICE)).float().sum().item()
                accuracy_list[f"rank_{rank}_bits_{bits}"] += correct
                data_iterator.set_description(f"Evaluating model, rank: {rank}, bits: {bits}")

    dataset_size = len(test_loader.dataset)
    for key in bit_stream_length_list:
        bit_stream_length_list[key] /= dataset_size
        accuracy_list[key] /= dataset_size

    save_results_to_csv(bit_stream_length_list,
                        accuracy_list,
                        csv_file=config["RDA"]["experience_configuration_csv_file_path"],
                        delimiter=config["RDA"]["csv_delimiter"])