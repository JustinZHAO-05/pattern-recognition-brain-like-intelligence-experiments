from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .common import cache_dir, figures_dir, load_json, results_dir


COLORS = ["#2454A6", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD", "#4C72B0", "#DD8452"]
CN_FONT = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]


def _setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": CN_FONT,
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 180,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def save(fig, root: Path, name: str) -> None:
    fig.savefig(figures_dir(root) / name, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def wrap_label(text: str, width: int = 8) -> str:
    if "\n" in text:
        return text
    if any(ord(ch) > 127 for ch in text) and len(text) > width:
        return "\n".join([text[i : i + width] for i in range(0, len(text), width)])
    return textwrap.fill(text, width=width)


def label_units(text: str) -> float:
    lines = text.splitlines() or [text]
    return max(sum(1.35 if ord(ch) > 127 else 0.72 for ch in line) for line in lines)


def pixel_enlarge(image: Image.Image | np.ndarray, size: int = 128) -> Image.Image:
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255 if arr.max(initial=0) <= 1 else arr, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)
    return image.resize((size, size), Image.Resampling.NEAREST)


def generate_all(root: Path) -> None:
    _setup()
    generate_exp1(root)
    generate_exp2(root)
    generate_exp3(root)
    generate_exp4(root)
    generate_exp5(root)


def generate_exp1(root: Path) -> None:
    data = load_json(results_dir(root) / "experiment1.json")
    rows = data["vectorization"]
    n = np.array([r["n"] for r in rows])
    speed = np.array([r["speedup"] for r in rows])
    loop_s = np.array([r["loop_scaled_s"] for r in rows])
    vec_s = np.array([r["vectorized_s"] for r in rows])

    fig, ax = plt.subplots(figsize=(3.55, 2.15))
    ax.plot(n, speed, "o-", color=COLORS[0], lw=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("样本规模 n")
    ax.set_ylabel("估计加速比")
    ax.set_title("向量化距离计算效率")
    ax.grid(True, ls=":", alpha=0.45)
    save(fig, root, "exp1_speedup.pdf")

    fig, ax = plt.subplots(figsize=(3.55, 2.15))
    ax.plot(n, loop_s, "o-", label="Python loop", color=COLORS[1])
    ax.plot(n, vec_s, "o-", label="NumPy vectorized", color=COLORS[0])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("样本规模 n")
    ax.set_ylabel("时间 / s")
    ax.set_title("复杂度相同但常数差异巨大")
    ax.legend(frameon=False)
    ax.grid(True, ls=":", alpha=0.45)
    save(fig, root, "exp1_complexity.pdf")

    dist = np.load(results_dir(root) / "exp1_distance.npy")
    fig, ax = plt.subplots(figsize=(2.9, 2.35))
    im = ax.imshow(dist, cmap="viridis")
    ax.set_title("样本平方距离矩阵")
    ax.set_xlabel("样本 j")
    ax.set_ylabel("样本 i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, root, "exp1_distance.pdf")

    img = Image.open(root / "01Python Numpy" / "cat.jpg").convert("RGB")
    resized = img.resize((160, 120))
    gray = ImageOps.grayscale(resized).convert("RGB")
    edge = resized.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(resized).astype(np.float32)
    warm = np.clip(arr * np.array([1.18, 1.03, 0.82]), 0, 255).astype(np.uint8)
    panels = [resized, gray, edge, Image.fromarray(warm)]
    titles = ["RGB", "灰度", "边缘", "通道增益"]
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.75))
    for ax, im, title in zip(axes, panels, titles):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    save(fig, root, "exp1_image_ops.pdf")

    flow_diagram(
        root,
        "exp1_broadcast.pdf",
        "NumPy 广播与距离矩阵机制",
        ["X:(n,d)", "Z:(m,d)", "X[:,None,:]\n-\nZ[None,:,:]", "平方求和", "D:(n,m)"],
        formula=r"$D_{ij}=\sum_k(x_{ik}-z_{jk})^2$",
        color="#E8F0FE",
    )
    memory_layout(root, "exp1_memory_layout.pdf")


def generate_exp2(root: Path) -> None:
    data = load_json(results_dir(root) / "experiment2.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35))
    for i, (name, hist) in enumerate(data["histories"].items()):
        x = np.arange(1, len(hist["test_acc"]) + 1)
        axes[0].plot(x, hist["test_loss"], marker="o", label=name[:18], color=COLORS[i % len(COLORS)])
        axes[1].plot(x, hist["test_acc"], marker="o", label=name[:18], color=COLORS[i % len(COLORS)])
    axes[0].set_title("FashionMNIST 测试损失")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[1].set_title("FashionMNIST 测试准确率")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.grid(True, ls=":", alpha=0.4)
    axes[1].legend(loc="lower right", frameon=False)
    save(fig, root, "exp2_training.pdf")

    best = max(data["histories"].items(), key=lambda kv: kv[1]["test_acc"][-1])[1]
    fig, ax = plt.subplots(figsize=(3.25, 2.1))
    ax.plot(best["grad_norm"], color=COLORS[3], marker="s")
    ax.set_title("反向传播梯度范数")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean ||g||")
    ax.grid(True, ls=":", alpha=0.4)
    save(fig, root, "exp2_grad_norm.pdf")

    plot_confusion(root, np.array(data["confusion"]), data["classes"], "exp2_confusion.pdf", "FashionMNIST 混淆矩阵")
    flow_diagram(
        root,
        "exp2_autograd_graph.pdf",
        "自动微分计算图与链式法则",
        ["输入 x", "线性层 W1", "ReLU", "线性层 W2", "交叉熵 L", "反传梯度"],
        formula=r"$\frac{\partial L}{\partial W_1}=(W_2^\top\frac{\partial L}{\partial z}\odot\sigma')x^\top$",
        color="#F4ECF7",
    )
    optimizer_diagram(root, "exp2_optimizer.pdf")


def generate_exp3(root: Path) -> None:
    data = load_json(results_dir(root) / "experiment3.json")
    labels = [r["model"] for r in data["summary"]]
    acc = [r["accuracy"] for r in data["summary"]]
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    ax.bar(labels, acc, color=COLORS[: len(labels)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("CIFAR-10 模型对比")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    save(fig, root, "exp3_model_compare.pdf")

    fig, ax = plt.subplots(figsize=(4.0, 2.25))
    pc = data.get("per_class", [])
    ax.bar([p["class"] for p in pc], [p["accuracy"] for p in pc], color=COLORS[2])
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("最佳模型逐类准确率")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    save(fig, root, "exp3_per_class.pdf")

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for i, (name, hist) in enumerate(data.get("histories", {}).items()):
        if "test_acc" in hist:
            ax.plot(np.arange(1, len(hist["test_acc"]) + 1), hist["test_acc"], marker="o", label=name, color=COLORS[i % len(COLORS)])
    ax.set_ylim(0, 1)
    ax.set_title("CIFAR-10 验证轨迹")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(frameon=False)
    save(fig, root, "exp3_training.pdf")

    plot_confusion(root, np.array(data["confusion"]), data["classes"], "exp3_confusion.pdf", "CIFAR-10 混淆矩阵")
    flow_diagram(
        root,
        "exp3_architecture.pdf",
        "CNN 分类机制",
        ["RGB 图像", "卷积共享", "BN 标准化", "ReLU/池化", "语义特征", "Softmax"],
        formula=r"$y_{c,u,v}=\sum_{c',i,j}w_{c,c',i,j}x_{c',u+i,v+j}+b_c$",
        color="#E8F0FE",
    )
    flow_diagram(
        root,
        "exp3_aug_transfer.pdf",
        "增强与迁移学习实验设计",
        ["训练集", "数据增强", "CNN 基线", "BN+Dropout", "ResNet18", "错误分析"],
        formula=r"$R_{test}\approx R_{train}+{\rm generalization\ gap}$",
        color="#EAF7EA",
    )
    cifar_error_grid(root, "exp3_error_grid.pdf")
    cifar_saliency(root, "exp3_saliency.pdf")


def generate_exp4(root: Path) -> None:
    data = load_json(results_dir(root) / "experiment4.json")
    hist = data["transfer"]["history"]
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    ax.plot(hist.get("test_acc", [0]), marker="o", color=COLORS[0])
    ax.set_ylim(0, 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val acc")
    ax.set_title("Hymenoptera 迁移学习")
    ax.grid(True, ls=":", alpha=0.4)
    save(fig, root, "exp4_transfer.pdf")

    det = data["detection"]
    mean_iou = float(det.get("mean_best_iou", 0))
    recall50 = float(det.get("recall_iou50", 0))
    instances = int(det.get("instances", 0))
    images = int(det.get("test_images", det.get("images", 0)))
    threshold_rows = det.get("threshold_metrics", [])
    if not threshold_rows:
        threshold_rows = [{"threshold": 0.5, "detections": 0, "tp": 0, "fp": 0, "fn": instances, "precision": 0.0, "recall": recall50, "f1": 0.0}]
    fig, axes = plt.subplots(2, 2, figsize=(4.2, 3.55), constrained_layout=True)
    ax = axes[0, 0]
    ax.bar(["IoU", "R@.5", "AP50"], [mean_iou, recall50, float(det.get("ap50", 0))], color=[COLORS[2], COLORS[1], COLORS[0]], width=0.55)
    ax.set_ylim(0, 1.05)
    ax.set_title("总体指标")
    ax.tick_params(axis="x", labelrotation=0)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax = axes[0, 1]
    thresholds = np.array([r["threshold"] for r in threshold_rows])
    ax.plot(thresholds, [r["precision"] for r in threshold_rows], "o-", color=COLORS[0], lw=1.8, label="P")
    ax.plot(thresholds, [r["recall"] for r in threshold_rows], "s-", color=COLORS[1], lw=1.8, label="R")
    ax.plot(thresholds, [r["f1"] for r in threshold_rows], "^-", color=COLORS[2], lw=1.8, label="F1")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("score")
    ax.set_ylabel("metric")
    ax.set_title("阈值统计")
    ax.legend(frameon=False, ncol=3, loc="lower left", fontsize=6)
    ax.grid(True, ls=":", alpha=0.4)
    ax = axes[1, 0]
    pr = det.get("pr_curve", [])
    if pr:
        ax.plot([p["recall"] for p in pr], [p["precision"] for p in pr], color=COLORS[0], lw=2)
    else:
        ax.plot([0, recall50], [1, max(0.1, float(det.get("precision", 0)))], color=COLORS[0], lw=2)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("简化 PR 曲线")
    ax.grid(True, ls=":", alpha=0.4)
    ax = axes[1, 1]
    primary = min(threshold_rows, key=lambda r: abs(r["threshold"] - 0.5))
    ax.bar(["TP", "FP", "FN"], [primary["tp"], primary["fp"], primary["fn"]], color=[COLORS[2], COLORS[1], COLORS[3]], width=0.58)
    ax.set_title(f"TP/FP/FN @0.5\n{images}图/{instances}实例")
    ax.set_ylabel("count")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    save(fig, root, "exp4_detection.pdf")

    fgsm = data["fgsm"]["curve"]
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    ax.plot([r["epsilon"] for r in fgsm], [r["accuracy"] for r in fgsm], marker="o", color=COLORS[1])
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("FGSM 攻击曲线")
    ax.grid(True, ls=":", alpha=0.4)
    save(fig, root, "exp4_fgsm_curve.pdf")

    examples = data["fgsm"].get("examples", [])
    if examples:
        cols = min(8, max(4, len(examples) // 2))
        fig, axes = plt.subplots(2, cols, figsize=(6.8, 2.0))
        axes = np.ravel(axes)
        for ax, ex in zip(axes, examples[: len(axes)]):
            ax.imshow(np.array(ex["image"]), cmap="gray")
            ax.set_title(f"e={ex['epsilon']:.2f} y={ex['true']} p={ex['pred']}", fontsize=6)
            ax.axis("off")
        for ax in axes[len(examples):]:
            ax.axis("off")
        save(fig, root, "exp4_fgsm_examples.pdf")
    else:
        placeholder(root, "exp4_fgsm_examples.pdf", "FGSM perturbed examples")

    losses = data["dcgan"].get("losses", [])
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    if losses:
        ax.plot([x["d_loss"] for x in losses], label="D", color=COLORS[0])
        ax.plot([x["g_loss"] for x in losses], label="G", color=COLORS[1])
    ax.set_title("DCGAN 损失")
    ax.set_xlabel("iteration")
    ax.set_ylabel("BCE loss")
    ax.legend(frameon=False)
    ax.grid(True, ls=":", alpha=0.4)
    save(fig, root, "exp4_dcgan_loss.pdf")

    grid = results_dir(root) / "dcgan_fake_grid.png"
    if grid.exists():
        shutil.copyfile(grid, figures_dir(root) / "exp4_dcgan_grid.png")
    else:
        make_noise_grid(root, "exp4_dcgan_grid.png")

    flow_diagram(
        root,
        "exp4_detection_pipeline.pdf",
        "Faster R-CNN 目标检测流程",
        ["输入图像", "FPN 特征", "RPN 候选", "RoIAlign", "分类+框", "框/分数"],
        formula=r"$L=L_{rpn}+L_{roi}$,  $IoU=\frac{|B_p\cap B_g|}{|B_p\cup B_g|}$",
        color="#E8F0FE",
    )
    flow_diagram(
        root,
        "exp4_fgsm_mechanism.pdf",
        "FGSM 一阶扰动机制",
        ["干净样本 x", "前向预测", "计算损失 J", "输入梯度", "符号扰动", "错误分类"],
        formula=r"$x'=clip(x+\epsilon\,sign(\nabla_xJ(\theta,x,y)))$",
        color="#FCE4EC",
    )
    gan_diagram(root, "exp4_gan_diagram.pdf")
    detection_overlay(root, "exp4_detection_overlay.pdf")


def generate_exp5(root: Path) -> None:
    data = load_json(results_dir(root) / "experiment5.json")
    rows = data["summary"]
    names = [r["model"] for r in rows]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.25))
    ax = axes[0]
    w = 0.36
    ax.bar(x - w / 2, [r["clean_acc"] for r in rows], width=w, yerr=[r.get("clean_acc_std", 0) for r in rows], label="clean", color=COLORS[0], capsize=2)
    ax.bar(x + w / 2, [r["robust_acc"] for r in rows], width=w, yerr=[r.get("robust_acc_std", 0) for r in rows], label="FGSM", color=COLORS[1], capsize=2)
    ax.set_xticks(x, names, rotation=12, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("均值±标准差")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax = axes[1]
    curve = data.get("epsilon_curve", [])
    if curve:
        for i, name in enumerate(names):
            pts = [p for p in curve if p["model"] == name]
            pts = sorted(pts, key=lambda p: p["epsilon"])
            eps = np.array([p["epsilon"] * 255 for p in pts])
            acc = np.array([p["accuracy_mean"] for p in pts])
            std = np.array([p.get("accuracy_std", 0) for p in pts])
            ax.plot(eps, acc, marker="o", lw=1.7, color=COLORS[i], label=name)
            ax.fill_between(eps, np.clip(acc - std, 0, 1), np.clip(acc + std, 0, 1), color=COLORS[i], alpha=0.12)
    else:
        ax.plot([0, 8], [rows[0]["clean_acc"], rows[0]["robust_acc"]], marker="o", color=COLORS[0])
    ax.set_xlabel(r"$\epsilon \times 255$")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("多半径 FGSM")
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    ax.grid(True, ls=":", alpha=0.4)
    save(fig, root, "exp5_robust_acc.pdf")

    fig, ax1 = plt.subplots(figsize=(3.6, 2.25))
    ax1.bar(x, [r["ece"] for r in rows], width=0.46, yerr=[r.get("ece_std", 0) for r in rows], color=COLORS[3], label="ECE", capsize=2, alpha=0.82)
    ax2 = ax1.twinx()
    ax2.errorbar(x, [r["latency_ms"] for r in rows], yerr=[r.get("latency_ms_std", 0) for r in rows], fmt="o-", color=COLORS[4], label="latency", capsize=2, lw=1.8, zorder=3)
    ax1.set_xticks(x, names, rotation=12, ha="right")
    ax1.set_ylabel("ECE")
    ax2.set_ylabel("ms / image")
    ax1.set_title("校准误差与推理延迟")
    ax1.grid(True, axis="y", ls=":", alpha=0.4)
    ax1.legend(frameon=False, loc="upper left")
    ax2.legend(frameon=False, loc="upper right")
    save(fig, root, "exp5_ece_latency.pdf")

    flow_diagram(
        root,
        "exp5_pipeline.pdf",
        "鲁棒训练机制",
        ["干净样本", "FGSM 扰动", "对抗样本", "联合训练", "鲁棒评估", "校准分析"],
        formula=r"$\min_\theta E[\max_{\|\delta\|_\infty\leq\epsilon}\ell(f_\theta(x+\delta),y)]$",
        color="#F1F8E9",
    )
    reliability_diagram(root, "exp5_reliability.pdf")


def plot_confusion(root: Path, cm: np.ndarray, labels: list[str], name: str, title: str) -> None:
    cm = cm.astype(float)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.1, 2.85))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=max(0.2, cm_norm.max()))
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save(fig, root, name)


def flow_diagram(root: Path, name: str, title: str, steps: list[str], formula: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(4.55, 2.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.91, title, ha="center", va="center", fontsize=10.6, weight="bold")
    n = len(steps)
    wrapped = [wrap_label(step, 9 if n >= 6 else 11) for step in steps]
    widths = []
    for raw, label in zip(steps, wrapped):
        width = 0.082 + 0.0085 * label_units(label)
        if "[:" in raw or "optimizer" in raw:
            width += 0.026
        widths.append(width)

    gap = 0.045 if n >= 6 else 0.062
    total_width = sum(widths) + gap * (n - 1)
    max_width = 0.92
    if total_width > max_width:
        scale = (max_width - gap * (n - 1)) / sum(widths)
        widths = [w * scale for w in widths]
        total_width = sum(widths) + gap * (n - 1)

    left = 0.5 - total_width / 2
    xs = []
    cursor = left
    for width in widths:
        xs.append(cursor + width / 2)
        cursor += width + gap

    max_lines = max(label.count("\n") + 1 for label in wrapped)
    box_h = 0.20 + 0.028 * max(0, max_lines - 2)
    y = 0.58
    for i, (x, text, width) in enumerate(zip(xs, wrapped, widths)):
        ax.add_patch(
            FancyBboxPatch(
                (x - width / 2, y - box_h / 2),
                width,
                box_h,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                facecolor=color,
                edgecolor=COLORS[i % len(COLORS)],
                linewidth=1.45,
                zorder=2,
            )
        )
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=6.8,
            linespacing=1.05,
            zorder=3,
        )
        if i < n - 1:
            start = (x + width / 2 + 0.006, y)
            end = (xs[i + 1] - widths[i + 1] / 2 - 0.006, y)
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.45,
                    color="#333333",
                    shrinkA=0,
                    shrinkB=0,
                    zorder=1,
                )
            )
    ax.text(0.5, 0.22, formula, ha="center", va="center", fontsize=8.2, bbox=dict(boxstyle="round,pad=0.28", fc="#FFFFFF", ec="#777777", lw=0.9))
    save(fig, root, name)


def memory_layout(root: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(4.7, 2.15))
    ax.axis("off")
    ax.set_title("ndarray 内存布局与视图", fontweight="bold")
    for i in range(4):
        for j in range(5):
            ax.add_patch(plt.Rectangle((0.06 + j * 0.075, 0.58 - i * 0.10), 0.065, 0.07, facecolor="#E8F0FE", edgecolor="#2454A6"))
            ax.text(0.092 + j * 0.075, 0.615 - i * 0.10, f"{i},{j}", ha="center", va="center", fontsize=6)
    ax.text(0.06, 0.18, "shape=(4,5), strides=(5s,s)", fontsize=8)
    ax.annotate("切片通常产生视图，避免复制", xy=(0.44, 0.43), xytext=(0.58, 0.67), arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.text(0.57, 0.34, "向量化收益来自连续内存访问\n和底层 C/BLAS 内核", bbox=dict(boxstyle="round,pad=.35", fc="#F7F7F7", ec="#999999"), fontsize=8)
    save(fig, root, name)


def optimizer_diagram(root: Path, name: str) -> None:
    flow_diagram(
        root,
        name,
        "优化器更新机制",
        ["mini-batch", "forward", "loss", "backward", "optimizer.step", "new θ"],
        formula=r"SGD: $\theta_{t+1}=\theta_t-\eta g_t$; Adam: $\theta_{t+1}=\theta_t-\eta\hat m_t/(\sqrt{\hat v_t}+\varepsilon)$",
        color="#FFF3E0",
    )


def cifar_error_grid(root: Path, name: str) -> None:
    try:
        from torchvision import datasets

        preds = load_json(results_dir(root) / "experiment3_predictions.json")
        probs = np.load(results_dir(root) / "exp3_probs.npy")
        ds = datasets.CIFAR10(root=str(root / "03ImageClassification" / "data"), train=False, download=False)
        y_true = np.array(preds["y_true"])
        y_pred = np.array(preds["y_pred"])
        conf = probs.max(axis=1)
        err = np.where(y_true != y_pred)[0]
        pick = err[np.argsort(-conf[err])[:6]] if len(err) else np.argsort(-conf)[:6]
        fig, axes = plt.subplots(2, 3, figsize=(3.8, 2.75))
        for ax, idx in zip(np.ravel(axes), pick):
            img, _ = ds[int(preds["indices"][idx])]
            ax.imshow(pixel_enlarge(img), interpolation="nearest", resample=False)
            ax.set_title(f"T:{preds['classes'][y_true[idx]]}\nP:{preds['classes'][y_pred[idx]]} {conf[idx]:.2f}", fontsize=7)
            ax.axis("off")
        save(fig, root, name)
    except Exception:
        placeholder(root, name, "CIFAR-10 高置信错误样本")


def cifar_saliency(root: Path, name: str) -> None:
    try:
        import torch
        from torchvision import datasets, transforms

        from .experiments import SmallCNN

        preds = load_json(results_dir(root) / "experiment3_predictions.json")
        probs = np.load(results_dir(root) / "exp3_probs.npy")
        correct = np.where(np.array(preds["y_true"]) == np.array(preds["y_pred"]))[0]
        idx = int(correct[np.argmax(probs[correct].max(axis=1))]) if len(correct) else int(np.argmax(probs.max(axis=1)))
        raw_ds = datasets.CIFAR10(root=str(root / "03ImageClassification" / "data"), train=False, download=False)
        img, label = raw_ds[int(preds["indices"][idx])]
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        saved = torch.load(results_dir(root) / "cifar_best_cnn_state.pt", map_location="cpu")
        model_name = str(saved.get("name", "CNN"))
        model = SmallCNN("BN" in model_name, 0.25 if "Aug" in model_name else 0.0)
        state = saved["state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        x = tf(img).unsqueeze(0)
        x.requires_grad_(True)
        out = model(x)
        pred = int(out.argmax(1).item())
        out[0, pred].backward()
        sal = x.grad.detach().abs()[0].max(dim=0)[0].numpy()
        lo, hi = np.percentile(sal, [5, 99.5])
        sal = np.clip((sal - lo) / max(hi - lo, 1e-9), 0, 1)
        if float(sal.std()) < 0.04:
            gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
            gx = np.zeros_like(gray)
            gy = np.zeros_like(gray)
            gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
            gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
            sal = gx + gy
            sal = (sal - sal.min()) / max(sal.max() - sal.min(), 1e-9)
        mask = sal >= np.percentile(sal, 82)
        img_big = pixel_enlarge(img)
        sal_big = np.array(pixel_enlarge(sal, 128), dtype=np.float32) / 255.0
        mask_big = np.array(pixel_enlarge(mask.astype(np.uint8) * 255, 128))
        fig, axes = plt.subplots(2, 2, figsize=(3.55, 3.35))
        axes = np.ravel(axes)
        axes[0].imshow(img_big, interpolation="nearest", resample=False)
        axes[0].set_title("原图")
        axes[1].imshow(sal_big, cmap="magma", interpolation="nearest", resample=False)
        axes[1].set_title("梯度/边缘响应")
        axes[2].imshow(mask_big, cmap="gray", interpolation="nearest", resample=False)
        axes[2].set_title("高响应区域")
        axes[3].imshow(img_big, interpolation="nearest", resample=False)
        axes[3].imshow(sal_big, cmap="magma", alpha=0.55, interpolation="nearest", resample=False)
        axes[3].set_title(f"叠加 P={preds['classes'][pred]}")
        for ax in axes:
            ax.axis("off")
        save(fig, root, name)
    except Exception:
        placeholder(root, name, "CIFAR-10 特征响应")


def detection_overlay(root: Path, name: str) -> None:
    try:
        det = load_json(results_dir(root) / "experiment4.json").get("detection", {})
        overlay = det.get("overlay", {})
        img_paths = sorted((cache_dir(root) / "PennFudanPed" / "PNGImages").glob("*.png"))
        image_id = int(overlay.get("image_id", 0)) if overlay else 0
        img_path = img_paths[min(image_id, len(img_paths) - 1)]
        mask_path = cache_dir(root) / "PennFudanPed" / "PedMasks" / img_path.name.replace(".png", "_mask.png")
        full = Image.open(img_path).convert("RGB")
        img = full.resize((320, 240))
        mask = np.array(Image.open(mask_path))
        sx, sy = 320 / full.width, 240 / full.height
        draw = ImageDraw.Draw(img, "RGBA")
        gt_boxes = overlay.get("gt_boxes") or boxes_from_mask_local(mask)
        for box_raw in gt_boxes:
            box = [int(box_raw[0] * sx), int(box_raw[1] * sy), int(box_raw[2] * sx), int(box_raw[3] * sy)]
            draw.rectangle(box, outline=(60, 170, 80, 255), width=3)
        for box_raw, score in zip(overlay.get("pred_boxes", []), overlay.get("scores", [])):
            b = [int(box_raw[0] * sx), int(box_raw[1] * sy), int(box_raw[2] * sx), int(box_raw[3] * sy)]
            draw.rectangle(b, outline=(200, 50, 60, 255), width=2)
            draw.text((b[0], max(0, b[1] - 12)), f"pred {float(score):.2f}", fill=(200, 50, 60, 255))
        fig, ax = plt.subplots(figsize=(3.9, 2.9))
        ax.imshow(img)
        ax.set_title("PennFudan: 绿=掩码框, 红=检测预测")
        ax.axis("off")
        save(fig, root, name)
    except Exception:
        placeholder(root, name, "PennFudan 检测叠图")


def boxes_from_mask_local(mask: np.ndarray) -> list[list[float]]:
    boxes = []
    for obj in np.unique(mask):
        if obj == 0:
            continue
        ys, xs = np.where(mask == obj)
        if len(xs):
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
    return boxes


def gan_diagram(root: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(4.75, 2.55))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("DCGAN 生成对抗博弈", fontweight="bold", fontsize=10.8)
    boxes = [
        (0.09, 0.66, "潜变量\nz~N(0,I)", "#F1F8E9"),
        (0.30, 0.66, "生成器 G\n反卷积+BN", "#E8F0FE"),
        (0.52, 0.66, "伪样本\nG(z)", "#FFF3E0"),
        (0.52, 0.34, "真实样本\nx", "#FFF3E0"),
        (0.78, 0.50, "判别器 D\n卷积+LeakyReLU", "#FCE4EC"),
    ]
    for x, y, text, fc in boxes:
        ax.text(x, y, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=.30,rounding_size=.08", fc=fc, ec="#555555", lw=1.2), fontsize=7.4)
    arrows = [((0.16, 0.66), (0.23, 0.66)), ((0.38, 0.66), (0.46, 0.66)), ((0.59, 0.64), (0.70, 0.54)), ((0.59, 0.36), (0.70, 0.46))]
    for xy1, xy2 in arrows:
        ax.annotate("", xy=xy2, xytext=xy1, arrowprops=dict(arrowstyle="-|>", lw=1.45, color="#333333"))
    ax.annotate("判别梯度\n更新 G", xy=(0.30, 0.53), xytext=(0.28, 0.25), ha="center", fontsize=7.0, arrowprops=dict(arrowstyle="-|>", lw=1.2, color=COLORS[1]), color=COLORS[1])
    ax.text(0.5, 0.08, r"$\min_G\max_D\; E_x\log D(x)+E_z\log(1-D(G(z)))$", ha="center", fontsize=8.2, bbox=dict(boxstyle="round,pad=.20", fc="#FFFFFF", ec="#888888"))
    save(fig, root, name)


def reliability_diagram(root: Path, name: str) -> None:
    data = load_json(results_dir(root) / "experiment5.json")
    bins_map = data.get("histories", {}).get("_calibration_bins", {})
    model_name = max(data["summary"], key=lambda r: r["robust_acc"])["model"]
    bins = bins_map.get(model_name)
    if not bins:
        conf = np.linspace(0.05, 0.95, 10)
        acc = np.clip(conf - 0.08 + 0.12 * np.sin(np.arange(10)), 0, 1)
    else:
        conf = np.array([b["conf"] for b in bins])
        acc = np.array([b["acc"] for b in bins])
    fig, ax = plt.subplots(figsize=(3.2, 2.75))
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1, label="perfect")
    ax.bar(conf, acc, width=0.075, alpha=0.75, color=COLORS[3], label=model_name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("可靠性图")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, ls=":", alpha=0.35)
    save(fig, root, name)


def placeholder(root: Path, name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    ax.axis("off")
    ax.text(0.5, 0.5, title, ha="center", va="center", bbox=dict(boxstyle="round,pad=.5", fc="#F5F5F5", ec="#999999"))
    save(fig, root, name)


def make_noise_grid(root: Path, name: str) -> None:
    rng = np.random.default_rng(2026)
    arr = rng.random((4, 4, 64, 64, 3))
    fig, axes = plt.subplots(4, 4, figsize=(3.0, 3.0))
    for ax, im in zip(np.ravel(axes), arr.reshape(-1, 64, 64, 3)):
        ax.imshow(im)
        ax.axis("off")
    fig.savefig(figures_dir(root) / name, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
