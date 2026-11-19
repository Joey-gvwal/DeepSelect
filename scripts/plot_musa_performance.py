"""Plot the README-aligned BF16 MUSA performance matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURE_SIZE_INCHES = (15, 4.4)
FIGURE_DPI = 150
DEEPSELECT_COLOR = "#66ccfe"
TORCH_TOPK_COLOR = "#ed0000"
REFERENCE_LEFT = 0.03543518518518519
REFERENCE_RIGHT = 0.99
REFERENCE_WSPACE = 0.042800275358415005


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text())["results"]
    batches = [6, 512, 4096]
    widths = [16384, 65536, 131072, 262144, 524288, 1048576]
    labels = ["16K", "64K", "128K", "256K", "512K", "1M"]

    figure, axes = plt.subplots(
        1, 3, figsize=FIGURE_SIZE_INCHES, sharey=True
    )
    x = np.arange(len(widths))
    bar_width = 0.38
    for axis, batch in zip(axes, batches):
        selected = {(row["batch"], row["width"]): row for row in rows}
        deepselect = [selected[(batch, width)]["deepselect_tbps"] for width in widths]
        torch_topk = [selected[(batch, width)]["torch_tbps"] for width in widths]
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
        axis.set_title(f"batch = {batch}")
        axis.set_xticks(x, labels, rotation=45)
        axis.set_xlabel("vocab_size", labelpad=1.6)
        axis.grid(axis="y", linestyle=":", linewidth=0.5)
    axes[0].set_ylabel("Effective bandwidth (TB/s)")
    axes[0].legend(loc="upper left")
    axes[0].set_ylim(0, 2)
    axes[0].set_yticks(np.arange(0, 2.01, 0.4))
    figure.suptitle(
        "bf16, topk = 512: DeepSelect vs torch.topk (MUSA)", y=0.93
    )
    figure.tight_layout()
    figure.subplots_adjust(
        left=REFERENCE_LEFT,
        right=REFERENCE_RIGHT,
        bottom=114 / 660,
        top=531.5 / 660,
        wspace=REFERENCE_WSPACE,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=FIGURE_DPI)


if __name__ == "__main__":
    main()
