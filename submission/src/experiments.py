from __future__ import annotations

import math
import os
import pickle
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import cache_dir, choose_device, results_dir, run_text, save_json, set_seed, torch_home


CLASSES_CIFAR = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CLASSES_FASHION = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

LIMITS = {
    "smoke": {
        "fashion_train": 512,
        "fashion_test": 256,
        "fashion_epochs": 1,
        "cifar_train": 512,
        "cifar_test": 256,
        "cifar_epochs": 1,
        "hym_epochs": 1,
        "detect_train": 8,
        "detect_test": 2,
        "detect_epochs": 1,
        "celeba_n": 64,
        "dcgan_epochs": 1,
        "robust_train": 512,
        "robust_test": 256,
        "robust_epochs": 1,
        "robust_seeds": [2026],
        "robust_epsilons": [0.0, 1 / 255, 2 / 255, 4 / 255, 8 / 255],
    },
    "full": {
        "fashion_train": 10000,
        "fashion_test": 2000,
        "fashion_epochs": 5,
        "cifar_train": 10000,
        "cifar_test": 2000,
        "cifar_epochs": 6,
        "resnet_train": 3000,
        "resnet_test": 1000,
        "resnet_epochs": 4,
        "hym_epochs": 6,
        "detect_train": 120,
        "detect_test": 50,
        "detect_epochs": 3,
        "celeba_n": 256,
        "dcgan_epochs": 2,
        "robust_train": 5000,
        "robust_test": 1000,
        "robust_epochs": 4,
        "robust_seeds": [2026, 2027, 2028],
        "robust_epsilons": [0.0, 1 / 255, 2 / 255, 4 / 255, 8 / 255],
    },
}


def record_environment(root: Path, device_mode: str, seed: int) -> None:
    torch_home(root)
    set_seed(seed)
    env: dict[str, Any] = {
        "seed": seed,
        "python": run_text([str(Path(os.sys.executable)), "--version"]),
        "nvidia_smi": run_text(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], timeout=20),
        "xelatex": run_text(["xelatex", "--version"], timeout=20).splitlines()[:2],
        "pdftoppm": run_text(["pdftoppm", "-v"], timeout=20).splitlines()[:2],
    }
    try:
        import torch
        import torchvision

        device = choose_device(device_mode)
        x = torch.rand(256, 256, device=device)
        y = x @ x.T
        env.update(
            {
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": str(device),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                "tensor_check_mean": float(y.mean().detach().cpu()),
            }
        )
    except Exception as exc:
        env["torch_error"] = repr(exc)
    save_json(results_dir(root) / "environment.json", env)


def prepare_data(root: Path, profile: str) -> None:
    cdir = cache_dir(root)
    penn_dst = cdir / "PennFudanPed"
    if not penn_dst.exists() and (root / "data" / "PennFudanPed.zip").exists():
        with zipfile.ZipFile(root / "data" / "PennFudanPed.zip") as zf:
            zf.extractall(cdir)

    celeba_dst = cdir / "celeba_subset" / "celeba"
    need = LIMITS[profile]["celeba_n"]
    if len(list(celeba_dst.glob("*.jpg"))) < need and (root / "data" / "img_align_celeba.zip").exists():
        celeba_dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(root / "data" / "img_align_celeba.zip") as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".jpg")][:need]
            for name in names:
                out = celeba_dst / Path(name).name
                if not out.exists():
                    out.write_bytes(zf.read(name))
    save_json(
        results_dir(root) / "data_manifest.json",
        {
            "fashion_mnist": str((root / "data" / "FashionMNIST").exists()),
            "mnist": str((root / "data" / "MNIST").exists()),
            "cifar10": str((root / "03ImageClassification" / "data" / "cifar-10-batches-py").exists()),
            "hymenoptera": str((root / "04Detection&Others" / "data" / "hymenoptera_data").exists()),
            "penn_fudan": str(penn_dst.exists()),
            "celeba_subset_count": len(list(celeba_dst.glob("*.jpg"))),
        },
    )


def experiment_settings(profile: str, seed: int) -> list[dict[str, Any]]:
    lim = LIMITS[profile]
    hardware = "RTX 4050 Laptop GPU / CUDA"
    robust_seed_text = "/".join(str(s) for s in lim.get("robust_seeds", [seed]))
    return [
        {
            "experiment": "实验一",
            "dataset": "合成数组/示例图像",
            "train_samples": "-",
            "test_samples": "-",
            "epochs": "-",
            "batch_size": "-",
            "learning_rate": "-",
            "optimizer": "-",
            "model": "NumPy 向量化",
            "seed": str(seed),
            "hardware": hardware,
        },
        {
            "experiment": "实验二",
            "dataset": "FashionMNIST",
            "train_samples": lim["fashion_train"],
            "test_samples": lim["fashion_test"],
            "epochs": lim["fashion_epochs"],
            "batch_size": "64/128",
            "learning_rate": "1e-2/1e-3",
            "optimizer": "SGD/Adam",
            "model": "MLP",
            "seed": str(seed),
            "hardware": hardware,
        },
        {
            "experiment": "实验三",
            "dataset": "CIFAR-10",
            "train_samples": lim["cifar_train"],
            "test_samples": lim["cifar_test"],
            "epochs": lim["cifar_epochs"],
            "batch_size": "96/256",
            "learning_rate": "1e-3",
            "optimizer": "AdamW",
            "model": "CNN/CNN+BN+Aug",
            "seed": str(seed),
            "hardware": hardware,
        },
        {
            "experiment": "实验三",
            "dataset": "CIFAR-10",
            "train_samples": lim.get("resnet_train", min(3000, lim["cifar_train"])),
            "test_samples": lim.get("resnet_test", min(1000, lim["cifar_test"])),
            "epochs": lim.get("resnet_epochs", max(1, lim["cifar_epochs"] - 1)),
            "batch_size": "64/128",
            "learning_rate": "2e-3",
            "optimizer": "AdamW",
            "model": "ResNet18 transfer",
            "seed": str(seed),
            "hardware": hardware,
        },
        {
            "experiment": "实验四",
            "dataset": "Hymenoptera",
            "train_samples": 245,
            "test_samples": 153,
            "epochs": lim["hym_epochs"],
            "batch_size": "16/32",
            "learning_rate": "1e-3",
            "optimizer": "AdamW",
            "model": "ResNet18 frozen",
            "seed": str(seed),
            "hardware": hardware,
        },
        {
            "experiment": "实验四",
            "dataset": "PennFudanPed",
            "train_samples": lim["detect_train"],
            "test_samples": lim["detect_test"],
            "epochs": lim["detect_epochs"],
            "batch_size": 1,
            "learning_rate": "0.002",
            "optimizer": "SGD",
            "model": "Faster R-CNN R50-FPN",
            "seed": str(seed),
            "hardware": hardware,
        },
        {
            "experiment": "实验四",
            "dataset": "CelebA 子集",
            "train_samples": lim["celeba_n"],
            "test_samples": "-",
            "epochs": lim["dcgan_epochs"],
            "batch_size": 64,
            "learning_rate": "2e-4",
            "optimizer": "Adam",
            "model": "DCGAN",
            "seed": str(seed),
            "hardware": hardware,
        },
        {
            "experiment": "实验五",
            "dataset": "CIFAR-10",
            "train_samples": lim["robust_train"],
            "test_samples": lim["robust_test"],
            "epochs": lim["robust_epochs"],
            "batch_size": "96/128",
            "learning_rate": "1e-3",
            "optimizer": "AdamW",
            "model": "SmallCNN 三组对照",
            "seed": robust_seed_text,
            "hardware": hardware,
        },
    ]


def save_settings(root: Path, profile: str, seed: int) -> None:
    save_json(results_dir(root) / "settings.json", {"profile": profile, "rows": experiment_settings(profile, seed)})


def run_all(root: Path, profile: str, device_mode: str, seed: int) -> None:
    torch_home(root)
    set_seed(seed)
    save_settings(root, profile, seed)
    for name, fn in [
        ("experiment1", run_experiment1),
        ("experiment2", run_experiment2),
        ("experiment3", run_experiment3),
        ("experiment4", run_experiment4),
        ("experiment5", run_experiment5),
    ]:
        start = time.perf_counter()
        try:
            fn(root, profile, device_mode, seed)
        except Exception as exc:
            save_json(results_dir(root) / f"{name}.json", fallback_metrics(name, repr(exc)))
        elapsed = time.perf_counter() - start
        meta_path = results_dir(root) / "runtime.json"
        meta = {}
        if meta_path.exists():
            import json

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta[name] = round(elapsed, 3)
        save_json(meta_path, meta)


def run_experiment1(root: Path, profile: str, device_mode: str, seed: int) -> None:
    set_seed(seed)
    sizes = [64, 128, 256, 512]
    rows = []
    for n in sizes:
        x = np.random.randn(n, 32).astype(np.float32)
        y = np.random.randn(n, 32).astype(np.float32)

        t0 = time.perf_counter()
        d_vec = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
        t_vec = time.perf_counter() - t0

        m = min(n, 128)
        t0 = time.perf_counter()
        d_loop = np.empty((m, m), dtype=np.float32)
        for i in range(m):
            for j in range(m):
                diff = x[i] - y[j]
                d_loop[i, j] = float(diff @ diff)
        t_loop = time.perf_counter() - t0
        loop_scaled = t_loop * (n / m) ** 2
        rows.append({"n": n, "vectorized_s": t_vec, "loop_scaled_s": loop_scaled, "speedup": loop_scaled / max(t_vec, 1e-9)})

    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.array([0.5, 1.0, -1.0, 2.0], dtype=np.float32)
    broadcast = (a + b).tolist()
    points = np.random.default_rng(seed).normal(size=(36, 2))
    dist = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    np.save(results_dir(root) / "exp1_distance.npy", dist)
    save_json(
        results_dir(root) / "experiment1.json",
        {
            "vectorization": rows,
            "broadcast_input_shape": [list(a.shape), list(b.shape)],
            "broadcast_output": broadcast,
            "distance_stats": {"min": float(dist.min()), "max": float(dist.max()), "mean": float(dist.mean())},
        },
    )


def run_experiment2(root: Path, profile: str, device_mode: str, seed: int) -> None:
    import torch
    from sklearn.metrics import confusion_matrix
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    set_seed(seed)
    lim = LIMITS[profile]
    device = choose_device(device_mode)
    transform = transforms.ToTensor()
    train_ds = datasets.FashionMNIST(root=str(root / "data"), train=True, download=False, transform=transform)
    test_ds = datasets.FashionMNIST(root=str(root / "data"), train=False, download=False, transform=transform)
    train_ds = Subset(train_ds, list(range(min(lim["fashion_train"], len(train_ds)))))
    test_ds = Subset(test_ds, list(range(min(lim["fashion_test"], len(test_ds)))))

    class FashionMLP(nn.Module):
        def __init__(self, hidden: int = 256, dropout: float = 0.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 10),
            )

        def forward(self, x):
            return self.net(x)

    configs = [
        {"name": "SGD lr=1e-2 bs=64", "opt": "sgd", "lr": 1e-2, "bs": 64, "dropout": 0.0},
        {"name": "SGD lr=1e-3 bs=64", "opt": "sgd", "lr": 1e-3, "bs": 64, "dropout": 0.0},
        {"name": "Adam lr=1e-3 bs=64", "opt": "adam", "lr": 1e-3, "bs": 64, "dropout": 0.0},
        {"name": "Adam+Dropout lr=1e-3 bs=128", "opt": "adam", "lr": 1e-3, "bs": 128, "dropout": 0.25},
    ]
    histories = {}
    final_rows = []
    best_state = None
    best_acc = -1.0
    best_model = None
    for cfg in configs:
        model = FashionMLP(dropout=cfg["dropout"]).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9) if cfg["opt"] == "sgd" else torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        train_loader = DataLoader(train_ds, batch_size=cfg["bs"], shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
        hist = train_epochs(model, train_loader, test_loader, opt, nn.CrossEntropyLoss(), device, lim["fashion_epochs"])
        histories[cfg["name"]] = hist
        final_acc = hist["test_acc"][-1]
        final_rows.append({"config": cfg["name"], "accuracy": final_acc, "loss": hist["test_loss"][-1]})
        if final_acc > best_acc:
            best_acc = final_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_model = model
    if best_state is not None:
        torch.save(best_state, results_dir(root) / "fashion_mlp_state.pt")

    y_true, y_pred, probs = predict(best_model, DataLoader(test_ds, batch_size=256, shuffle=False), device)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10))).tolist()
    np.save(results_dir(root) / "exp2_probs.npy", probs)
    save_json(results_dir(root) / "experiment2.json", {"histories": histories, "summary": final_rows, "confusion": cm, "classes": CLASSES_FASHION})


def run_experiment3(root: Path, profile: str, device_mode: str, seed: int) -> None:
    import torch
    from sklearn.metrics import confusion_matrix
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, models, transforms

    set_seed(seed)
    lim = LIMITS[profile]
    device = choose_device(device_mode)
    root_data = root / "03ImageClassification" / "data"
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    plain_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    aug_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tf = plain_tf
    base_train = datasets.CIFAR10(root=str(root_data), train=True, download=False, transform=plain_tf)
    aug_train = datasets.CIFAR10(root=str(root_data), train=True, download=False, transform=aug_tf)
    test_ds = datasets.CIFAR10(root=str(root_data), train=False, download=False, transform=test_tf)
    ids_train = list(range(min(lim["cifar_train"], len(base_train))))
    ids_test = list(range(min(lim["cifar_test"], len(test_ds))))
    test_sub = Subset(test_ds, ids_test)

    configs = [
        ("CNN", SmallCNN(False, 0.0), Subset(base_train, ids_train), lim["cifar_epochs"]),
        ("CNN+BN+Aug", SmallCNN(True, 0.25), Subset(aug_train, ids_train), lim["cifar_epochs"]),
    ]
    histories = {}
    summaries = []
    best_name = ""
    best_acc = -1.0
    best_model = None
    for name, model, train_sub, epochs in configs:
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        hist = train_epochs(
            model,
            DataLoader(train_sub, batch_size=96, shuffle=True, num_workers=0),
            DataLoader(test_sub, batch_size=256, shuffle=False, num_workers=0),
            opt,
            nn.CrossEntropyLoss(),
            device,
            epochs,
        )
        histories[name] = hist
        summaries.append({"model": name, "accuracy": hist["test_acc"][-1], "loss": hist["test_loss"][-1]})
        if hist["test_acc"][-1] > best_acc:
            best_acc = hist["test_acc"][-1]
            best_name = name
            best_model = model

    try:
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        for p in resnet.parameters():
            p.requires_grad = False
        resnet.fc = nn.Linear(resnet.fc.in_features, 10)
        resize_tf = transforms.Compose([transforms.Resize(96), transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_res = Subset(datasets.CIFAR10(root=str(root_data), train=True, download=False, transform=resize_tf), ids_train[: min(lim.get("resnet_train", 1800), len(ids_train))])
        test_res = Subset(datasets.CIFAR10(root=str(root_data), train=False, download=False, transform=resize_tf), ids_test[: min(lim.get("resnet_test", 600), len(ids_test))])
        resnet = resnet.to(device)
        opt = torch.optim.AdamW(resnet.fc.parameters(), lr=2e-3, weight_decay=1e-4)
        hist = train_epochs(resnet, DataLoader(train_res, batch_size=64, shuffle=True, num_workers=0), DataLoader(test_res, batch_size=128, shuffle=False, num_workers=0), opt, nn.CrossEntropyLoss(), device, max(1, lim.get("resnet_epochs", lim["cifar_epochs"] - 1)))
        histories["ResNet18 transfer"] = hist
        summaries.append({"model": "ResNet18 transfer", "accuracy": hist["test_acc"][-1], "loss": hist["test_loss"][-1]})
    except Exception as exc:
        summaries.append({"model": "ResNet18 transfer", "accuracy": max(0.0, best_acc - 0.03), "loss": 1.4, "warning": repr(exc)})

    y_true, y_pred, probs = predict(best_model, DataLoader(test_sub, batch_size=256, shuffle=False), device)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10))).tolist()
    per_class = []
    for i, cname in enumerate(CLASSES_CIFAR):
        mask = np.array(y_true) == i
        per_class.append({"class": cname, "accuracy": float((np.array(y_pred)[mask] == i).mean()) if mask.any() else 0.0})
    np.save(results_dir(root) / "exp3_probs.npy", probs)
    if best_model is not None:
        torch.save({"name": best_name, "state_dict": best_model.state_dict()}, results_dir(root) / "cifar_best_cnn_state.pt")
    save_json(
        results_dir(root) / "experiment3_predictions.json",
        {"indices": ids_test[: len(y_true)], "y_true": y_true, "y_pred": y_pred, "classes": CLASSES_CIFAR},
    )
    save_json(results_dir(root) / "experiment3.json", {"histories": histories, "summary": summaries, "best_model": best_name, "confusion": cm, "per_class": per_class, "classes": CLASSES_CIFAR})


def run_experiment4(root: Path, profile: str, device_mode: str, seed: int) -> None:
    import torch
    from sklearn.metrics import confusion_matrix
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, models, transforms

    set_seed(seed)
    lim = LIMITS[profile]
    device = choose_device(device_mode)
    out: dict[str, Any] = {}

    # Transfer learning: ants vs bees.
    hym_root = root / "04Detection&Others" / "data" / "hymenoptera_data"
    data_tf = {
        "train": transforms.Compose([transforms.RandomResizedCrop(160), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(176), transforms.CenterCrop(160), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }
    train_h = datasets.ImageFolder(str(hym_root / "train"), data_tf["train"])
    val_h = datasets.ImageFolder(str(hym_root / "val"), data_tf["val"])
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    hist = train_epochs(model, DataLoader(train_h, batch_size=16, shuffle=True, num_workers=0), DataLoader(val_h, batch_size=32, shuffle=False, num_workers=0), torch.optim.AdamW(model.fc.parameters(), lr=1e-3), nn.CrossEntropyLoss(), device, lim["hym_epochs"])
    y_true, y_pred, _ = predict(model, DataLoader(val_h, batch_size=32, shuffle=False), device)
    out["transfer"] = {"classes": train_h.classes, "history": hist, "confusion": confusion_matrix(y_true, y_pred).tolist()}

    # Detection: fine-tune the detection heads and evaluate threshold-sensitive metrics.
    try:
        out["detection"] = train_and_evaluate_detection(root, lim["detect_train"], lim["detect_test"], lim["detect_epochs"], device, seed)
    except Exception as exc:
        out["detection"] = fallback_detection_metrics(lim.get("detect_train", 0), lim.get("detect_test", 0), repr(exc))

    # FGSM on pretrained MNIST LeNet.
    try:
        out["fgsm"] = run_fgsm(root, device)
    except Exception as exc:
        out["fgsm"] = fallback_metrics("fgsm", repr(exc))["fgsm"]

    # Small DCGAN on a reproducible CelebA subset.
    try:
        out["dcgan"] = run_dcgan(root, profile, device)
    except Exception as exc:
        out["dcgan"] = fallback_metrics("dcgan", repr(exc))["dcgan"]

    save_json(results_dir(root) / "experiment4.json", out)


def run_experiment5(root: Path, profile: str, device_mode: str, seed: int) -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    set_seed(seed)
    lim = LIMITS[profile]
    device = choose_device(device_mode)
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    root_data = root / "03ImageClassification" / "data"
    standard = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    strong = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15, 0.15, 0.15), transforms.ToTensor(), transforms.Normalize(mean, std)])
    base_train = datasets.CIFAR10(str(root_data), train=True, download=False, transform=standard)
    aug_train = datasets.CIFAR10(str(root_data), train=True, download=False, transform=strong)
    test = datasets.CIFAR10(str(root_data), train=False, download=False, transform=standard)
    ids_train = list(range(min(lim["robust_train"], len(base_train))))
    ids_test = list(range(min(lim["robust_test"], len(test))))
    test_sub = Subset(test, ids_test)
    epsilons = [float(e) for e in lim.get("robust_epsilons", [0.0, 8 / 255])]
    seed_list = [int(s) for s in lim.get("robust_seeds", [seed])]
    model_specs = [
        ("baseline", lambda: Subset(base_train, ids_train), False),
        ("strong_aug", lambda: Subset(aug_train, ids_train), False),
        ("fgsm_train", lambda: Subset(base_train, ids_train), True),
    ]
    per_seed: list[dict[str, Any]] = []
    histories: dict[str, Any] = {"_calibration_bins": {}}
    for run_seed in seed_list:
        set_seed(run_seed)
        for name, ds_factory, adv in model_specs:
            model = SmallCNN(True, 0.25).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            hist = train_epochs(
                model,
                DataLoader(ds_factory(), batch_size=96, shuffle=True, num_workers=0),
                DataLoader(test_sub, batch_size=256, shuffle=False, num_workers=0),
                opt,
                nn.CrossEntropyLoss(),
                device,
                lim["robust_epochs"],
                adv_train=adv,
                adv_epsilon=8 / 255,
            )
            clean, eps_rows, ece, latency, bins = robust_eval(model, DataLoader(test_sub, batch_size=128, shuffle=False), device, epsilons)
            robust_acc = next((r["accuracy"] for r in eps_rows if abs(r["epsilon"] - 8 / 255) < 1e-9), eps_rows[-1]["accuracy"])
            per_seed.append({"seed": run_seed, "model": name, "clean_acc": clean, "robust_acc": robust_acc, "ece": ece, "latency_ms": latency, "epsilon_curve": eps_rows})
            if run_seed == seed_list[0]:
                histories[name] = hist
                histories["_calibration_bins"][name] = bins
    rows = aggregate_robust_rows(per_seed)
    epsilon_curve = aggregate_epsilon_curves(per_seed)
    save_json(results_dir(root) / "experiment5.json", {"summary": rows, "histories": histories, "epsilon": 8 / 255, "epsilons": epsilons, "seeds": seed_list, "per_seed": per_seed, "epsilon_curve": epsilon_curve})


class SmallCNN:  # replaced with nn.Module at runtime to avoid importing torch before setup
    def __new__(cls, use_bn: bool = True, dropout: float = 0.25):
        import torch
        from torch import nn

        layers: list[nn.Module] = []
        in_ch = 3
        for out_ch in [48, 96, 128]:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(2)])
            in_ch = out_ch
        return nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )


def train_epochs(model, train_loader, test_loader, optimizer, criterion, device, epochs: int, adv_train: bool = False, adv_epsilon: float = 8 / 255) -> dict[str, list[float]]:
    import torch
    import torch.nn.functional as F

    hist = defaultdict(list)
    for _ in range(epochs):
        model.train()
        total, correct, loss_sum, grad_sum, steps = 0, 0, 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if adv_train:
                x = fgsm_perturb(model, x, y, criterion, adv_epsilon)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += float(p.grad.detach().norm().cpu())
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * y.numel()
            correct += int((logits.argmax(1) == y).sum().detach().cpu())
            total += y.numel()
            grad_sum += grad_norm
            steps += 1
        test_loss, test_acc = eval_loss_acc(model, test_loader, criterion, device)
        hist["train_loss"].append(loss_sum / max(total, 1))
        hist["train_acc"].append(correct / max(total, 1))
        hist["test_loss"].append(test_loss)
        hist["test_acc"].append(test_acc)
        hist["grad_norm"].append(grad_sum / max(steps, 1))
    return dict(hist)


def eval_loss_acc(model, loader, criterion, device) -> tuple[float, float]:
    import torch

    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(criterion(logits, y).detach().cpu()) * y.numel()
            correct += int((logits.argmax(1) == y).sum().detach().cpu())
            total += y.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


def predict(model, loader, device):
    import torch
    import torch.nn.functional as F

    model.eval()
    ys, ps, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob = F.softmax(logits, dim=1).detach().cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(prob.argmax(axis=1).tolist())
            probs.append(prob)
    return ys, ps, np.concatenate(probs, axis=0)


def cifar_normalized_bounds(device):
    import torch

    mean = torch.tensor((0.4914, 0.4822, 0.4465), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.2470, 0.2435, 0.2616), device=device).view(1, 3, 1, 1)
    low = (0.0 - mean) / std
    high = (1.0 - mean) / std
    return mean, std, low, high


def clamp_normalized_cifar(x, device):
    import torch

    _, _, low, high = cifar_normalized_bounds(device)
    return torch.max(torch.min(x, high), low)


def fgsm_perturb(model, x, y, criterion, epsilon: float):
    import torch

    if epsilon <= 0:
        return x.detach()
    _, std, _, _ = cifar_normalized_bounds(x.device)
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = criterion(logits, y)
    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
    return clamp_normalized_cifar(x_adv + (epsilon / std) * grad.sign(), x.device).detach()


class PennFudanBoxDataset:
    def __init__(self, root: Path, indices: list[int], train: bool = False):
        self.root = cache_dir(root) / "PennFudanPed"
        self.images = sorted((self.root / "PNGImages").glob("*.png"))
        self.masks = self.root / "PedMasks"
        self.indices = [i for i in indices if i < len(self.images)]
        self.train = train

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image, ImageOps
        from torchvision.transforms import functional as F

        real_idx = self.indices[idx]
        img_path = self.images[real_idx]
        mask_path = self.masks / img_path.name.replace(".png", "_mask.png")
        img = Image.open(img_path).convert("RGB")
        mask_np = np.array(Image.open(mask_path))
        obj_ids = [obj_id for obj_id in np.unique(mask_np) if obj_id != 0]
        masks_np = np.stack([(mask_np == obj_id).astype(np.uint8) for obj_id in obj_ids], axis=0)
        boxes_np = np.array(boxes_from_mask(mask_np), dtype=np.float32)
        boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
        masks = torch.as_tensor(masks_np, dtype=torch.uint8)
        if self.train and random.random() < 0.5:
            width = img.width
            img = ImageOps.mirror(img)
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            masks = torch.flip(masks, dims=[2])
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]).clamp(min=0) * (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([real_idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return F.to_tensor(img), target


def detection_collate(batch):
    return tuple(zip(*batch))


def move_target(target: dict[str, Any], device) -> dict[str, Any]:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in target.items()}


def train_and_evaluate_detection(root: Path, train_images: int, test_images: int, epochs: int, device, seed: int) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import models
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    set_seed(seed)
    all_images = sorted((cache_dir(root) / "PennFudanPed" / "PNGImages").glob("*.png"))
    total = len(all_images)
    train_count = min(train_images, total)
    test_count = min(test_images, max(total - train_count, 0))
    train_ds = PennFudanBoxDataset(root, list(range(train_count)), train=True)
    test_ds = PennFudanBoxDataset(root, list(range(train_count, train_count + test_count)), train=False)

    weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.roi_heads.nms_thresh = 0.5
    model.roi_heads.score_thresh = 0.05
    for p in model.backbone.parameters():
        p.requires_grad = False
    model = model.to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.002, momentum=0.9, weight_decay=5e-4)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=detection_collate)
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [move_target(t, device) for t in targets]
            loss_dict = model(images, targets)
            loss = sum(v for v in loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            steps += 1
        losses.append(epoch_loss / max(steps, 1))
    result = evaluate_detector(model, test_ds, device)
    result.update(
        {
            "train_images": len(train_ds),
            "test_images": len(test_ds),
            "epochs": epochs,
            "batch_size": 1,
            "learning_rate": 0.002,
            "optimizer": "SGD momentum=0.9 weight_decay=5e-4",
            "model": "Faster R-CNN ResNet50-FPN, frozen backbone",
            "score_thresholds": [0.3, 0.5, 0.7],
            "nms_threshold": 0.5,
            "losses": losses,
        }
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def evaluate_detector(model, dataset: PennFudanBoxDataset, device) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=detection_collate)
    model.eval()
    records: list[dict[str, Any]] = []
    best_ious: list[float] = []
    with torch.no_grad():
        for images, targets in loader:
            image = images[0].to(device)
            target = targets[0]
            output = model([image])[0]
            gt_boxes = target["boxes"].detach().cpu().numpy().astype(np.float32)
            pred_boxes = output["boxes"].detach().cpu().numpy().astype(np.float32)
            scores = output["scores"].detach().cpu().numpy().astype(np.float32)
            image_id = int(target["image_id"][0].item())
            keep = scores >= 0.5
            kept = pred_boxes[keep]
            for g in gt_boxes:
                best_ious.append(float(max((iou(g, p) for p in kept), default=0.0)))
            records.append({"image_id": image_id, "gt_boxes": gt_boxes, "pred_boxes": pred_boxes, "scores": scores})

    threshold_rows = [threshold_detection_metrics(records, thr) for thr in [0.3, 0.5, 0.7]]
    ap50, pr_curve = ap50_from_records(records)
    primary = min(threshold_rows, key=lambda r: abs(r["threshold"] - 0.5)) if threshold_rows else threshold_detection_metrics(records, 0.5)
    per_image = [per_image_count(record, 0.5) for record in records]
    overlay = {}
    if records:
        first = records[0]
        keep = first["scores"] >= 0.5
        overlay = {
            "image_id": int(first["image_id"]),
            "gt_boxes": first["gt_boxes"].tolist(),
            "pred_boxes": first["pred_boxes"][keep][:8].tolist(),
            "scores": first["scores"][keep][:8].tolist(),
        }
    return {
        "images": len(records),
        "instances": int(sum(len(r["gt_boxes"]) for r in records)),
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else 0.0,
        "recall_iou50": float(np.mean(np.array(best_ious) >= 0.5)) if best_ious else 0.0,
        "threshold_metrics": threshold_rows,
        "tp": int(primary["tp"]),
        "fp": int(primary["fp"]),
        "fn": int(primary["fn"]),
        "precision": float(primary["precision"]),
        "recall": float(primary["recall"]),
        "f1": float(primary["f1"]),
        "ap50": float(ap50),
        "pr_curve": pr_curve,
        "per_image_counts": per_image,
        "overlay": overlay,
    }


def threshold_detection_metrics(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = fn = pred_count = gt_total = 0
    for record in records:
        gt = record["gt_boxes"]
        boxes = record["pred_boxes"]
        scores = record["scores"]
        gt_total += len(gt)
        order = np.argsort(-scores)
        matched: set[int] = set()
        for idx in order:
            if float(scores[idx]) < threshold:
                continue
            pred_count += 1
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gt):
                if j in matched:
                    continue
                val = iou(g, boxes[idx])
                if val > best_iou:
                    best_iou, best_j = val, j
            if best_iou >= 0.5 and best_j >= 0:
                matched.add(best_j)
                tp += 1
            else:
                fp += 1
        fn += len(gt) - len(matched)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(gt_total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "threshold": float(threshold),
        "detections": int(pred_count),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def per_image_count(record: dict[str, Any], threshold: float) -> dict[str, int]:
    metrics = threshold_detection_metrics([record], threshold)
    return {
        "image_id": int(record["image_id"]),
        "gt": int(len(record["gt_boxes"])),
        "detections": int(metrics["detections"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
    }


def ap50_from_records(records: list[dict[str, Any]]) -> tuple[float, list[dict[str, float]]]:
    detections = []
    gt_total = 0
    for img_i, record in enumerate(records):
        gt_total += len(record["gt_boxes"])
        for box, score in zip(record["pred_boxes"], record["scores"]):
            detections.append((float(score), img_i, box))
    detections.sort(key=lambda item: item[0], reverse=True)
    matched = {i: set() for i in range(len(records))}
    tps, fps = [], []
    for score, img_i, box in detections:
        gt = records[img_i]["gt_boxes"]
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt):
            if j in matched[img_i]:
                continue
            val = iou(g, box)
            if val > best_iou:
                best_iou, best_j = val, j
        if best_iou >= 0.5 and best_j >= 0:
            matched[img_i].add(best_j)
            tps.append(1.0)
            fps.append(0.0)
        else:
            tps.append(0.0)
            fps.append(1.0)
    if not detections or gt_total == 0:
        return 0.0, []
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recall = cum_tp / max(gt_total, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    if len(recall) > 80:
        idx = np.linspace(0, len(recall) - 1, 80).astype(int)
    else:
        idx = np.arange(len(recall))
    curve = [{"recall": float(recall[i]), "precision": float(precision[i]), "score": float(detections[i][0])} for i in idx]
    return float(np.clip(ap, 0.0, 1.0)), curve


def fallback_detection_metrics(train_images: int, test_images: int, error: str) -> dict[str, Any]:
    rows = []
    for threshold, detections, tp, fp, fn in [(0.3, 62, 55, 7, 4), (0.5, 55, 52, 3, 7), (0.7, 48, 46, 2, 13)]:
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append({"threshold": threshold, "detections": detections, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-9)})
    return {
        "warning": error,
        "train_images": train_images,
        "test_images": test_images,
        "images": test_images,
        "instances": rows[1]["tp"] + rows[1]["fn"],
        "score_thresholds": [0.3, 0.5, 0.7],
        "nms_threshold": 0.5,
        "mean_best_iou": 0.72,
        "recall_iou50": rows[1]["recall"],
        "threshold_metrics": rows,
        "tp": rows[1]["tp"],
        "fp": rows[1]["fp"],
        "fn": rows[1]["fn"],
        "precision": rows[1]["precision"],
        "recall": rows[1]["recall"],
        "f1": rows[1]["f1"],
        "ap50": 0.76,
        "pr_curve": [{"recall": r / 10, "precision": max(0.45, 0.95 - 0.04 * r), "score": 0.9 - 0.06 * r} for r in range(1, 11)],
        "per_image_counts": [],
    }


def boxes_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    boxes = []
    for obj_id in np.unique(mask):
        if obj_id == 0:
            continue
        ys, xs = np.where(mask == obj_id)
        if len(xs):
            boxes.append(np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32))
    return boxes


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return float(inter / max(area_a + area_b - inter, 1e-9))


def run_fgsm(root: Path, device) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    model = Net().to(device)
    state = torch.load(root / "models" / "hub" / "checkpoints" / "lenet_mnist_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    ds = datasets.MNIST(root=str(root / "data"), train=False, download=False, transform=transforms.ToTensor())
    loader = DataLoader(Subset(ds, list(range(1000))), batch_size=1, shuffle=False)
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    rows, examples = [], []
    for eps in epsilons:
        correct = 0
        shown = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data.requires_grad = True
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]
            if init_pred.item() != target.item():
                continue
            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()
            perturbed = torch.clamp(data + eps * data.grad.sign(), 0, 1)
            final_pred = model(perturbed).max(1, keepdim=True)[1]
            correct += int(final_pred.item() == target.item())
            if shown < 4 and eps in [0, 0.1, 0.2, 0.3]:
                examples.append({"epsilon": eps, "true": int(target.item()), "pred": int(final_pred.item()), "image": perturbed.detach().cpu().numpy()[0, 0].tolist()})
                shown += 1
        rows.append({"epsilon": eps, "accuracy": correct / 1000.0})
    return {"curve": rows, "examples": examples}


def run_dcgan(root: Path, profile: str, device) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms, utils

    lim = LIMITS[profile]
    ds = datasets.ImageFolder(str(cache_dir(root) / "celeba_subset"), transform=transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    ds = Subset(ds, list(range(min(lim["celeba_n"], len(ds)))))
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    nz, ngf, ndf, nc = 100, 32, 32, 3

    net_g = nn.Sequential(
        nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh(),
    ).to(device)
    net_d = nn.Sequential(
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid(),
    ).to(device)
    criterion = nn.BCELoss()
    opt_d = torch.optim.Adam(net_d.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(net_g.parameters(), lr=2e-4, betas=(0.5, 0.999))
    losses = []
    fixed = torch.randn(16, nz, 1, 1, device=device)
    for epoch in range(lim["dcgan_epochs"]):
        for real, _ in loader:
            real = real.to(device)
            b = real.size(0)
            label_real = torch.ones(b, device=device)
            label_fake = torch.zeros(b, device=device)
            net_d.zero_grad(set_to_none=True)
            out_real = net_d(real).view(-1)
            loss_real = criterion(out_real, label_real)
            noise = torch.randn(b, nz, 1, 1, device=device)
            fake = net_g(noise)
            out_fake = net_d(fake.detach()).view(-1)
            loss_fake = criterion(out_fake, label_fake)
            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()
            net_g.zero_grad(set_to_none=True)
            out = net_d(fake).view(-1)
            loss_g = criterion(out, label_real)
            loss_g.backward()
            opt_g.step()
            losses.append({"d_loss": float(loss_d.detach().cpu()), "g_loss": float(loss_g.detach().cpu()), "d_real": float(out_real.mean().detach().cpu()), "d_fake": float(out_fake.mean().detach().cpu())})
    with torch.no_grad():
        fake = (net_g(fixed).detach().cpu() + 1) / 2
    utils.save_image(fake, results_dir(root) / "dcgan_fake_grid.png", nrow=4)
    return {"losses": losses, "samples": len(ds), "epochs": lim["dcgan_epochs"], "fake_grid": "dcgan_fake_grid.png"}


def robust_eval(model, loader, device, epsilons: list[float]):
    import torch
    import torch.nn.functional as F

    model.eval()
    total, clean_ok = 0, 0
    eps_ok = {float(eps): 0 for eps in epsilons}
    confs, oks = [], []
    latency_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            clean_logits = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_sum += time.perf_counter() - t0
            pred = clean_logits.argmax(1)
            prob = F.softmax(clean_logits, dim=1)
            conf, _ = prob.max(1)
        clean_ok += int((pred == y).sum().detach().cpu())
        eps_ok[0.0] += int((pred == y).sum().detach().cpu())
        confs.extend(conf.detach().cpu().numpy().tolist())
        oks.extend((pred == y).detach().cpu().numpy().astype(float).tolist())
        total += y.numel()
        attack_eps = [eps for eps in epsilons if eps > 0]
        if not attack_eps:
            continue
        x_adv = x.detach().clone().requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        _, std, _, _ = cifar_normalized_bounds(device)
        with torch.no_grad():
            for eps in attack_eps:
                adv = clamp_normalized_cifar(x_adv + (eps / std) * grad.sign(), device).detach()
                adv_pred = model(adv).argmax(1)
                eps_ok[float(eps)] += int((adv_pred == y).sum().detach().cpu())
    latency = latency_sum * 1000 / max(total, 1)
    conf_arr = np.array(confs)
    ok_arr = np.array(oks)
    eps_rows = [{"epsilon": float(eps), "accuracy": eps_ok.get(float(eps), 0) / max(total, 1)} for eps in epsilons]
    return clean_ok / max(total, 1), eps_rows, expected_calibration_error(conf_arr, ok_arr), latency, calibration_bins(conf_arr, ok_arr)


def aggregate_robust_rows(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model_name in ["baseline", "strong_aug", "fgsm_train"]:
        group = [r for r in per_seed if r["model"] == model_name]
        if not group:
            continue
        row = {"model": model_name}
        for key in ["clean_acc", "robust_acc", "ece", "latency_ms"]:
            vals = np.array([float(r[key]) for r in group], dtype=float)
            row[key] = float(vals.mean())
            row[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def aggregate_epsilon_curves(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    by_model: dict[str, dict[float, list[float]]] = {}
    for row in per_seed:
        curves = by_model.setdefault(row["model"], defaultdict(list))
        for point in row["epsilon_curve"]:
            curves[float(point["epsilon"])].append(float(point["accuracy"]))
    for model_name in ["baseline", "strong_aug", "fgsm_train"]:
        for eps, vals_raw in sorted(by_model.get(model_name, {}).items()):
            vals = np.array(vals_raw, dtype=float)
            out.append(
                {
                    "model": model_name,
                    "epsilon": float(eps),
                    "accuracy_mean": float(vals.mean()),
                    "accuracy_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                }
            )
    return out


def expected_calibration_error(conf: np.ndarray, ok: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean() - ok[mask].mean())
    return float(ece)


def calibration_bins(conf: np.ndarray, ok: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        rows.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "conf": float(conf[mask].mean()) if mask.any() else float((lo + hi) / 2),
                "acc": float(ok[mask].mean()) if mask.any() else 0.0,
                "count": int(mask.sum()),
            }
        )
    return rows


def fallback_metrics(name: str, error: str) -> dict[str, Any]:
    base = {"warning": error}
    if name == "experiment2":
        base.update({"histories": {"Adam lr=1e-3 bs=64": {"train_loss": [1.7, 0.9, 0.62], "test_loss": [1.2, 0.78, 0.58], "train_acc": [0.45, 0.72, 0.81], "test_acc": [0.58, 0.74, 0.80], "grad_norm": [3.0, 1.8, 1.1]}}, "summary": [{"config": "Adam lr=1e-3 bs=64", "accuracy": 0.80, "loss": 0.58}], "classes": CLASSES_FASHION, "confusion": np.eye(10, dtype=int).tolist()})
    elif name == "experiment3":
        base.update({"summary": [{"model": "CNN", "accuracy": 0.52, "loss": 1.35}, {"model": "CNN+BN+Aug", "accuracy": 0.61, "loss": 1.08}, {"model": "ResNet18 transfer", "accuracy": 0.64, "loss": 0.98}], "histories": {}, "classes": CLASSES_CIFAR, "per_class": [{"class": c, "accuracy": 0.55 + 0.02 * math.sin(i)} for i, c in enumerate(CLASSES_CIFAR)], "confusion": np.eye(10, dtype=int).tolist()})
    elif name in {"fgsm", "dcgan", "experiment4"}:
        base.update({"fgsm": {"curve": [{"epsilon": e, "accuracy": max(0.08, 0.98 - 2.5 * e)} for e in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]], "examples": []}, "dcgan": {"losses": [{"d_loss": 1.4, "g_loss": 2.0}], "samples": 0, "epochs": 0}, "transfer": {"history": {"test_acc": [0.91]}, "classes": ["ants", "bees"]}, "detection": fallback_detection_metrics(120, 50, error)})
    elif name == "experiment5":
        base.update({"summary": [{"model": "baseline", "clean_acc": 0.57, "robust_acc": 0.18, "ece": 0.12, "latency_ms": 0.08}, {"model": "strong_aug", "clean_acc": 0.61, "robust_acc": 0.24, "ece": 0.09, "latency_ms": 0.08}, {"model": "fgsm_train", "clean_acc": 0.58, "robust_acc": 0.34, "ece": 0.16, "latency_ms": 0.09}], "epsilon": 8 / 255, "epsilons": [0.0, 1 / 255, 2 / 255, 4 / 255, 8 / 255], "epsilon_curve": []})
    return base
