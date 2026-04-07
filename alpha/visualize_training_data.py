"""
训练数据可视化脚本

用于查看 alpha/training_data 下的 .pkl 训练样本内容（state, policy, value）。
"""

import argparse
import os
import pickle
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class Sample:
    state: np.ndarray
    policy: np.ndarray
    value: float


def _default_data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _as_sample(obj) -> Sample:
    if not isinstance(obj, (tuple, list)) or len(obj) != 3:
        raise ValueError(f"样本不是三元组: type={type(obj)}")
    state, policy, value = obj
    state = np.asarray(state)
    policy = np.asarray(policy)
    value = _safe_float(value)
    return Sample(state=state, policy=policy, value=value)


def load_pkl(path: str) -> list[Sample]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError(f"pkl 顶层不是 list: {path}")
    return [_as_sample(x) for x in data]


def _policy_stats(policy: np.ndarray) -> tuple[float, float, int]:
    p = np.asarray(policy, dtype=np.float64).reshape(-1)
    mass = float(np.sum(p[p > 0]))
    if mass > 0:
        q = p / mass
        q = q[q > 0]
        entropy = float(-np.sum(q * np.log(q + 1e-12)))
        top_idx = int(np.argmax(p))
        return mass, entropy, top_idx
    return mass, 0.0, -1


def plot_sample(sample: Sample, title: str) -> None:
    s = sample.state
    p = sample.policy
    if s.ndim != 3 or s.shape[0] < 9:
        raise ValueError(f"state 形状不符合预期: {tuple(s.shape)}")
    n = int(s.shape[-1])
    my = (s[0] > 0.5).astype(np.int32)
    opp = (s[8] > 0.5).astype(np.int32)
    board = my + 2 * opp

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax0, ax1 = axes

    cmap = plt.get_cmap("gray_r", 3)
    im0 = ax0.imshow(board, cmap=cmap, vmin=0, vmax=2)
    ax0.set_title("棋盘(X=当前方, O=对手)")
    ax0.set_xticks(range(n))
    ax0.set_yticks(range(n))
    for r in range(n + 1):
        ax0.axhline(r - 0.5, color="black", linewidth=0.5)
        ax0.axvline(r - 0.5, color="black", linewidth=0.5)
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    if p.size == n * n:
        heat = p.reshape(n, n)
    elif p.size == 25 and n == 5:
        heat = p.reshape(5, 5)
    else:
        heat = np.zeros((n, n), dtype=np.float32)
    im1 = ax1.imshow(heat, cmap="viridis")
    ax1.set_title("policy 热力图")
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    mass, ent, top_idx = _policy_stats(p)
    if top_idx >= 0 and p.size:
        top_r, top_c = divmod(top_idx, n)
        top_txt = f"top=({top_r},{top_c}) p={float(np.max(p)):.4f}"
    else:
        top_txt = "top=<none>"

    fig.suptitle(f"{title} | value={sample.value:.3f} | mass={mass:.3f} | H={ent:.3f} | {top_txt}", fontsize=10)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def resolve_pkl_path(pkl_arg: str, data_dir: str) -> str:
    if os.path.isabs(pkl_arg):
        path = pkl_arg
    else:
        path = os.path.join(data_dir, pkl_arg)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"pkl 文件不存在: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="指定一个 pkl 文件并弹窗可视化其中样本")
    parser.add_argument("--data-dir", default=_default_data_dir(), help="训练数据目录")
    parser.add_argument("--pkl", required=True, help="目标 pkl 文件名或绝对路径")
    parser.add_argument("--sample-index", type=int, default=0, help="显示该 pkl 中第几个样本(从0开始)")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    pkl_path = resolve_pkl_path(args.pkl, data_dir)
    samples = load_pkl(pkl_path)
    if not samples:
        print(f"文件为空: {pkl_path}")
        return 0

    idx = int(args.sample_index)
    if idx < 0 or idx >= len(samples):
        raise SystemExit(f"sample-index 越界: {idx}, 可选范围 [0, {len(samples) - 1}]")

    s = samples[idx]
    print(f"文件: {pkl_path}")
    print(f"样本总数: {len(samples)}")
    print(f"当前样本: {idx}")
    print(f"state shape: {tuple(s.state.shape)}")
    print(f"policy shape: {tuple(s.policy.shape)}")
    print(f"value: {s.value:.3f}")
    plot_sample(s, title=f"{os.path.basename(pkl_path)} | sample={idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
