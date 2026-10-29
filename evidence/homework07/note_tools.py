"""Supplied helpers for HW07 Task 07: load the note's cut-out letter crops, prepare them to look like
archive images, read them with a trained model, and score the reading against the checked transcription.

You don't need to modify (or fully understand) this file. The notebook's Task 07 description explains
what each step does and why.
"""
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def crop_key(path):
    """Sort crops by page, then letter number (so letter_100 comes after letter_11)."""
    page, letter = re.findall(r'\d+', Path(path).stem)
    return int(page), int(letter)


def load_crops(folder):
    """Return the crop paths in reading order and the naive 28 x 28 grayscale batch."""
    paths = sorted(Path(folder).glob('*.png'), key=crop_key)
    raw = np.stack([cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (28, 28)) for p in paths])
    return paths, torch.from_numpy(raw).float().unsqueeze(1) / 255.0


def extract_ink(path):
    image = cv2.imread(str(path))
    height, width = image.shape[:2]
    if height < 6 or width < 6:
        return np.zeros((height, width), np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    page_colored = ((hsv[..., 1] < 40) & (hsv[..., 2] > 200)).astype(np.uint8)
    _, regions = cv2.connectedComponents(page_colored)
    border = set(np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]]).tolist()) - {0}
    paper_region = ~np.isin(regions, list(border)) if border else np.ones((height, width), bool)
    if paper_region.sum() < 10:
        return np.zeros((height, width), np.float32)
    colors = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    _, groups, _ = cv2.kmeans(colors[paper_region], 2, None,
                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 5, cv2.KMEANS_PP_CENTERS)
    group_map = np.full((height, width), -1)
    group_map[paper_region] = groups.ravel()
    paper_edge = paper_region & (cv2.erode(paper_region.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
    paper_group = np.bincount(group_map[paper_edge], minlength=2).argmax()
    ink = group_map == 1 - paper_group
    if ink.sum() > 0.6 * paper_region.sum():
        ink = group_map == paper_group
    count, pieces, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    touching_edge = set(pieces[paper_edge & ink].tolist())
    cleaned = np.zeros((height, width), np.float32)
    for piece in range(1, count):
        if piece not in touching_edge and stats[piece, cv2.CC_STAT_AREA] >= 0.02 * paper_region.sum():
            cleaned[pieces == piece] = 1
    return cleaned if cleaned.any() else ink.astype(np.float32)


def to_emnist_style(mask, size=24):
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return np.zeros((28, 28), np.float32)
    box = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    h, w = box.shape
    side = max(h, w)
    square = np.zeros((side, side), np.float32)
    top, left = (side - h) // 2, (side - w) // 2
    square[top:top + h, left:left + w] = box
    small = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    out = np.zeros((28, 28), np.float32)
    offset = (28 - size) // 2
    out[offset:offset + size, offset:offset + size] = small
    return out


def prepare_crops(paths):
    """Separate each letter from its paper and frame it like an archive character."""
    return torch.from_numpy(np.stack([to_emnist_style(extract_ink(p)) for p in paths])).unsqueeze(1)


def load_truth(path):
    """The checked transcription with spaces removed (the crops don't include spaces)."""
    return ''.join(line.replace(' ', '') for line in Path(path).read_text().split('\n') if line)


def edit_distance(a, b):
    """Fewest single-character insertions, deletions, and substitutions turning a into b."""
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        diagonal, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (ca != cb))
    return row[-1]


def character_accuracy(reading, truth):
    """1 - edit distance / length of the truth, ignoring case."""
    return 1 - edit_distance(reading.upper(), truth.upper()) / len(truth)


@torch.no_grad()
def read_crops(model, batch, classes):
    """The model's best guess for every crop, as one string."""
    if isinstance(model, nn.Module):
        model.eval()
    logits = model(batch) if isinstance(model, nn.Module) else model.forward(batch)
    return ''.join(classes[i] for i in logits.argmax(dim=1).tolist())


def show_preparation(paths, raw, prepared, model, classes, show=(0, 4, 6, 9, 18, 23, 44, 61)):
    """Draw a few crops at each stage with the model's raw and prepared readings."""
    fig, axes = plt.subplots(3, len(show), figsize=(len(show) * 1.4, 4.6))
    for column, index in enumerate(show):
        axes[0, column].imshow(cv2.cvtColor(cv2.imread(str(paths[index])), cv2.COLOR_BGR2RGB))
        axes[1, column].imshow(raw[index, 0], cmap='gray', vmin=0, vmax=1)
        axes[2, column].imshow(prepared[index, 0], cmap='gray', vmin=0, vmax=1)
        before = read_crops(model, raw[index:index + 1], classes)
        after = read_crops(model, prepared[index:index + 1], classes)
        axes[2, column].set_xlabel(f'{before} \u2192 {after}', fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for row, label in enumerate(['crop', 'raw input', 'prepared']):
        axes[row, 0].set_ylabel(label)
    fig.suptitle('The same crops before and after preparation (CNN reading: raw \u2192 prepared)')
    fig.tight_layout()
    plt.show()
