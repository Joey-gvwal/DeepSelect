#!/usr/bin/env python3
"""Plot the README-aligned FP32 MUSA sampling benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURE_SIZE_INCHES = (7.2, 4.6)
FIGURE_DPI = 150
DEEPSELECT_COLOR = "#66ccfe"
TORCH_TOPK_COLOR = "#ed0000"
REFERENCE_LEFT = 80 / 1080
REFERENCE_RIGHT = 1058 / 1080
REFERENCE_BOTTOM = 114 / 690
REFERENCE_TOP = 637 / 690


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text())["results"]
    batches = [6, 256, 512, 768, 4096]
    selected = {row["batch"]: row for row in rows}
    deepselect = [selected[batch]["deepselect_tbps"] for batch in batches]
    torch_topk = [selected[batch]["torch_tbps"] for batch in batches]

    figure, axis = plt.subplots(figsize=FIGURE_SIZE_INCHES)
    x = np.arange(len(batches))
    bar_width = 0.38
    axis.bar(
        x - bar_width / 2,
        deepselect,
        bar_width,
        label="DeepSelect",
        color=DEEPSELECT_COLOR,
    )
    axis.bar(
        x + bar_width / 2,
        torch_topk,
        bar_width,
        label="torch.topk",
        color=TORCH_TOPK_COLOR,
    )
    axis.set_title(
        "fp32, vocab_size = 129280, topk = 512: "
        "DeepSelect vs torch.topk (MUSA)"
    )
    axis.set_xticks(x, [str(batch) for batch in batches], rotation=45)
    axis.set_xlabel("batch_size", labelpad=1.6)
    axis.set_ylabel("Effective bandwidth (TB/s)")
    axis.set_ylim(0, 1)
    axis.set_yticks(np.arange(0, 1.01, 0.2))
    axis.grid(axis="y", linestyle=":", linewidth=0.5)
    axis.legend(loc="upper right")
    figure.subplots_adjust(
        left=REFERENCE_LEFT,
        right=REFERENCE_RIGHT,
        bottom=REFERENCE_BOTTOM,
        top=REFERENCE_TOP,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=FIGURE_DPI)


if __name__ == "__main__":
    main()
