#!/usr/bin/env python3
"""Generate lab03_solutions.ipynb — PyTorch fundamentals lab prepping for HW04."""
import json

def md(source):
    lines = source.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}

def code(source):
    lines = source.strip("\n").split("\n")
    return {"cell_type": "code", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]], "outputs": [], "execution_count": None}

cells = []

# ═══ CELL: Title ═══
cells.append(md("""# CSE 30124 - Introduction to Artificial Intelligence: Lab 03 (5 pts.)

- NETID:

This assignment covers the following topics:
- Creating and manipulating PyTorch tensors
- Converting between NumPy arrays and PyTorch tensors
- Tensor reshaping for neural network input formats
- Defining models with `nn.Module`
- Loading data with `Dataset` and `DataLoader`
- The PyTorch training loop (`zero_grad`, `backward`, `step`)

It will consist of 7 tasks:

| Task ID  | Description                                      | Points |
|----------|--------------------------------------------------|--------|
| 00       | Setup                                            | 0      |
| 01       | NumPy ↔ Tensor Conversion                       | 1      |
| 02       | Prepare Images for a CNN                         | 1      |
| 03       | Define a Small Classifier                        | 1      |
| 04       | Create a DataLoader                              | 1      |
| 05       | Train and Evaluate                               | 1      |
| 06       | Generate Police Report                           | 0      |

Please complete all sections. Some questions will require written answers, while others will involve coding. Be sure to run your code cells to verify your solutions."""))

# ═══ CELL: Story Intro ═══
cells.append(md("""## *Story Progression*

Your image segmentation work paid off — the forensics team now has individual letter images extracted from the ransom notes. But identifying what character each letter actually is? That's a job for a neural network.

<div class="thumbnail">
    <img src="https://williamtheisen.com/nd-cse-30124-homeworks/imgs/lab03/ransom_note.png" class="img-responsive"/>
    <figcaption style="text-align: center"><b>Evidence 1: Letters extracted from the ransom note — but what do they say?</b></figcaption>
</div>

Director Bryant tells you the department has a character recognition system, but it runs on **PyTorch** — a deep learning framework you haven't used before. Before you can feed the evidence through the classifier, you need to learn how PyTorch represents and processes data. Detective Gaff slides a PyTorch tutorial across the table. "Better start reading," he says."""))

# ═══ CELL: Task 00 Header ═══
cells.append(md("""## Task 00: Setup (0 pts.)
### Task 00: Code (0 pts.)"""))

# ═══ CELL: Task 00 Code ═══
cells.append(code("""import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import google.colab
    REPO_URL = "https://github.com/nd-cse-30124-fa25/cse-30124-homeworks.git"
    REPO_NAME = "cse-30124-homeworks"
    LAB_FOLDER = "labs/lab03"

    %cd /content/
    if not os.path.exists(REPO_NAME):
        !git clone {REPO_URL}

    %cd {REPO_NAME}/{LAB_FOLDER}

except ImportError:
    print("Not running on Colab - assuming local setup.")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version:   {np.__version__}")

# Device selection (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device:    {device}")
print("Setup complete!")"""))

# ═══════════════════════════════════════════════════════════════
# TASK 01: NumPy ↔ Tensor Conversion
# ═══════════════════════════════════════════════════════════════
cells.append(md("""## Task 01: NumPy ↔ Tensor Conversion (1 pt.)

The EMNIST dataset used by the forensics classifier is stored as NumPy `.npz` files. Load the training data, convert the images and labels to PyTorch tensors, and verify the conversion.

Steps:
1. Load the `.npz` file with `np.load()`
2. Extract the `X` (images) and `y` (labels) arrays
3. Flatten images from `(n, 28, 28)` to `(n, 784)` and normalize pixel values to `[0, 1]`
4. Convert images to a `float32` tensor and labels to a `long` tensor (required by PyTorch's loss functions)
5. Print shapes, dtypes, and pixel value range

### Task 01: Code (1 pt.)"""))

# ═══ CELL: Task 01 Code ═══
cells.append(code("""# TODO: Load the EMNIST training data
data = np.load("../../evidence/homework04/emnist_balanced_small/emnist_balanced_small_train.npz")
images_np = data['X']       # shape: (n_samples, 28, 28), dtype: uint8
labels_np = data['y']       # shape: (n_samples,), dtype: int64

print(f"NumPy images: shape={images_np.shape}, dtype={images_np.dtype}")
print(f"NumPy labels: shape={labels_np.shape}, dtype={labels_np.dtype}")

# TODO: Convert to PyTorch tensors with correct dtypes
# Flatten images to (n_samples, 784) — the FFN classifier expects flat vectors
# Also normalize pixel values from [0, 255] to [0.0, 1.0]
images_tensor = torch.tensor(images_np.reshape(-1, 784), dtype=torch.float32) / 255.0
labels_tensor = torch.tensor(labels_np, dtype=torch.long)

print(f"\\nTensor images: shape={images_tensor.shape}, dtype={images_tensor.dtype}")
print(f"Tensor labels: shape={labels_tensor.shape}, dtype={labels_tensor.dtype}")
print(f"Pixel value range: [{images_tensor.min():.1f}, {images_tensor.max():.1f}]")

# TODO: Verify conversion — check shapes and first few labels
print(f"\\nFirst 10 labels: {labels_tensor[:10]}")"""))

# ═══ CELL: Task 01 Expected Output ═══
cells.append(md("""### Task 01: Expected Output (1 pt.)
```
NumPy images: shape=(9400, 28, 28), dtype=uint8
NumPy labels: shape=(9400,), dtype=int64

Tensor images: shape=torch.Size([9400, 784]), dtype=torch.float32
Tensor labels: shape=torch.Size([9400]), dtype=torch.int64
Pixel value range: [0.0, 1.0]

First 10 labels: tensor([...])
```

*Note:* Images are stored as 28×28 pixel arrays but we flatten them to 784-dimensional vectors for the FFN classifier. We also normalize from `[0, 255]` to `[0, 1]` — neural networks train better on small values. Labels are integers 0-46 corresponding to the 47 EMNIST character classes. PyTorch requires `float32` for model inputs and `long` (int64) for classification targets."""))

# ═══ CELL: Story Progression ═══
cells.append(md("""### *Story Progression*

Good — the evidence data is now in PyTorch tensor format. But Detective Gaff points out that the forensics classifier is a **convolutional neural network**, which expects images in a very specific shape: `(batch, channels, height, width)`. Your current data is flat 784-dimensional vectors (we flattened them for the FFN). Time to learn how to reshape tensors between different formats..."""))

# ═══════════════════════════════════════════════════════════════
# TASK 02: Prepare Images for a CNN
# ═══════════════════════════════════════════════════════════════
cells.append(md("""## Task 02: Prepare Images for a CNN (1 pt.)

The forensics CNN expects input shaped `(batch_size, 1, 28, 28)` — batch of single-channel 28×28 images. Your data is currently `(n_samples, 784)` flat vectors. Reshape the entire dataset.

Steps:
1. Reshape `images_tensor` from `(n_samples, 784)` to `(n_samples, 28, 28)` using `.reshape()`
2. Add a channel dimension to get `(n_samples, 1, 28, 28)` using `.unsqueeze()`
3. Print the shape at each step
4. Verify by displaying a grid of the first 8 images

### Task 02: Code (1 pt.)"""))

# ═══ CELL: Task 02 Code ═══
cells.append(code("""# TODO: Reshape from (n_samples, 784) to (n_samples, 28, 28)
images_2d = images_tensor.reshape(-1, 28, 28)
print(f"Step 1 — flat to 2D: {images_tensor.shape} -> {images_2d.shape}")

# TODO: Add channel dimension to get (n_samples, 1, 28, 28)
images_cnn = images_2d.unsqueeze(1)
print(f"Step 2 — add channel: {images_2d.shape} -> {images_cnn.shape}")

# TODO: Display first 8 images in a grid
fig, axes = plt.subplots(1, 8, figsize=(16, 2))
for i in range(8):
    axes[i].imshow(images_cnn[i, 0].numpy(), cmap='gray')
    axes[i].set_title(f"Label: {labels_tensor[i].item()}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()"""))

# ═══ CELL: Task 02 Expected Output ═══
cells.append(md("""### Task 02: Expected Output (1 pt.)
```
Step 1 — flat to 2D: torch.Size([<n>, 784]) -> torch.Size([<n>, 28, 28])
Step 2 — add channel: torch.Size([<n>, 28, 28]) -> torch.Size([<n>, 1, 28, 28])
```
Plus a row of 8 grayscale EMNIST character images with their labels. The images should be recognizable as handwritten letters/digits.

*Note:* The `(batch, channels, height, width)` format is standard for PyTorch CNNs. In HW04, you'll feed data in exactly this shape."""))

# ═══ CELL: Story Progression ═══
cells.append(md("""### *Story Progression*

"The data's in the right shape now," says Director Bryant. "But we need an actual classifier to process it. The old one crashed — can you build a new one?"

You've seen classifiers before (kNN, SVM), but neural networks are different. In PyTorch, you define a model as a Python class. Time to learn the `nn.Module` pattern..."""))

# ═══════════════════════════════════════════════════════════════
# TASK 03: Define a Small Classifier
# ═══════════════════════════════════════════════════════════════
cells.append(md("""## Task 03: Define a Small Classifier (1 pt.)

Build a simple single-layer classifier for the 47-class EMNIST dataset:

- Input: 784 features (28×28 flattened image)
- Output layer: 47 units (one per class, no activation)

Steps:
1. Define the class inheriting from `nn.Module`
2. Create one `nn.Linear` layer in `__init__`
3. Return its output in `forward()`
4. Create an instance and run a forward pass on a small batch to verify

### Task 03: Code (1 pt.)"""))

# ═══ CELL: Task 03 Code ═══
cells.append(code("""# TODO: Define the classifier
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 47)

    def forward(self, x):
        return self.fc(x)

# TODO: Create an instance and test with a small batch
model = SimpleClassifier()
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\\nTotal parameters: {total_params:,}")

# TODO: Forward pass on first 4 samples
test_batch = images_tensor[:4]           # (4, 784)
output = model(test_batch)
print(f"\\nInput shape:  {test_batch.shape}")
print(f"Output shape: {output.shape}")

# Get predicted classes
predictions = torch.argmax(output, dim=1)
print(f"Predicted classes: {predictions}")
print(f"(These are random — the model hasn't been trained yet!)")"""))

# ═══ CELL: Task 03 Expected Output ═══
cells.append(md("""### Task 03: Expected Output (1 pt.)
```
SimpleClassifier(
  (fc): Linear(in_features=784, out_features=47, bias=True)
)

Total parameters: 36,895

Input shape:  torch.Size([4, 784])
Output shape: torch.Size([4, 47])
Predicted classes: tensor([...])
(These are random — the model hasn't been trained yet!)
```

*Note:* The output is 47 raw "logits" (one per class). The predicted class is the index of the largest logit (`torch.argmax`). Since the weights are random, the predictions are meaningless — but the shapes are correct! In HW04, you'll define similar architectures for both FFN and CNN classifiers."""))

# ═══ CELL: Story Progression ═══
cells.append(md("""### *Story Progression*

"We've got a model," says Detective Gaff, "but it doesn't know anything yet — it's just guessing." He's right. The model needs to be **trained** on the EMNIST data. But with thousands of images, you can't feed them all at once. You need to organize the data into **batches**..."""))

# ═══════════════════════════════════════════════════════════════
# TASK 04: Create a DataLoader
# ═══════════════════════════════════════════════════════════════
cells.append(md("""## Task 04: Create a DataLoader (1 pt.)

Create a proper training pipeline from the full EMNIST dataset:

1. Create a `TensorDataset` from `images_tensor` and `labels_tensor`
2. Split into training (80%) and validation (20%) sets using `random_split`
3. Create `DataLoader`s for each — training with `shuffle=True`, validation with `shuffle=False`
4. Iterate over one training batch and print its shapes

### Task 04: Code (1 pt.)"""))

# ═══ CELL: Task 04 Code ═══
cells.append(code("""from torch.utils.data import random_split

# TODO: Create TensorDataset
full_dataset = TensorDataset(images_tensor, labels_tensor)
print(f"Full dataset size: {len(full_dataset)}")

# TODO: Split into train (80%) and val (20%)
n_total = len(full_dataset)
n_train = int(0.8 * n_total)
n_val = n_total - n_train
train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# TODO: Create DataLoaders with batch_size=64
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# TODO: Print one batch shape
for batch_x, batch_y in train_loader:
    print(f"\\nTraining batch: x={batch_x.shape}, y={batch_y.shape}")
    break

for batch_x, batch_y in val_loader:
    print(f"Validation batch: x={batch_x.shape}, y={batch_y.shape}")
    break"""))

# ═══ CELL: Task 04 Expected Output ═══
cells.append(md("""### Task 04: Expected Output (1 pt.)
```
Full dataset size: <n_total>
Train: <n_train>, Val: <n_val>

Training batch: x=torch.Size([64, 784]), y=torch.Size([64])
Validation batch: x=torch.Size([64, 784]), y=torch.Size([64])
```

*Note:* In HW04, you'll also implement a custom `Dataset` class that loads images from PNG files on disk. The `__len__` and `__getitem__` pattern is the same."""))

# ═══ CELL: Story Progression ═══
cells.append(md("""### *Story Progression*

"Data's organized, model's defined," Director Bryant counts on his fingers. "Now we just need to... teach it?" That's training — the process of adjusting the model's weights by showing it examples and correcting its mistakes. The core of deep learning is a simple 3-step loop..."""))

# ═══════════════════════════════════════════════════════════════
# TASK 05: Train and Evaluate
# ═══════════════════════════════════════════════════════════════
cells.append(md("""## Task 05: Train and Evaluate (1 pt.)

Train the `SimpleClassifier` from Task 03 on the training data and evaluate on the validation data.

Steps:
1. Create an `Adam` optimizer with `lr=0.001` and a `CrossEntropyLoss` criterion
2. Train for **5 epochs**, printing the average loss each epoch
3. After training, evaluate on the validation set: compute accuracy using `torch.argmax` and `torch.no_grad()`

### Task 05: Code (1 pt.)"""))

# ═══ CELL: Task 05 Code ═══
cells.append(code("""# TODO: Create optimizer and loss
model = SimpleClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# TODO: Training loop — 5 epochs
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/5: avg loss = {avg_loss:.4f}")

# TODO: Evaluate on validation set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_x, batch_y in val_loader:
        logits = model(batch_x)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

accuracy = correct / total
print(f"\\nValidation accuracy: {accuracy:.4f} ({correct}/{total})")"""))

# ═══ CELL: Task 05 Expected Output ═══
cells.append(md("""### Task 05: Expected Output (1 pt.)
```
Epoch 1/5: avg loss = <decreasing>
Epoch 2/5: avg loss = <decreasing>
Epoch 3/5: avg loss = <decreasing>
Epoch 4/5: avg loss = <decreasing>
Epoch 5/5: avg loss = <decreasing>

Validation accuracy: <value> (<correct>/<total>)
```

The loss should **decrease** each epoch — this means the model is learning! The accuracy won't be great after just 5 epochs with a simple model, but it should be well above random chance (1/47 ≈ 2%).

*Note:* In HW04 you'll train for more epochs, use larger models (including CNNs), and implement these same patterns both from scratch and with PyTorch's built-in tools."""))

# ═══ CELL: Final Story ═══
cells.append(md("""### *Story Progression*

The classifier is learning! Even after just 5 epochs, it's doing far better than random chance. You've now mastered all the core PyTorch skills you'll need for **Homework 04**:

1. **Tensors** — creating, converting from NumPy, moving between devices
2. **Reshape/permute/unsqueeze** — preparing data for FFN and CNN input formats
3. **`nn.Module`** — defining model architectures with `__init__` and `forward`
4. **`Dataset` and `DataLoader`** — organizing data into shuffled mini-batches
5. **The training loop** — `zero_grad()` → `backward()` → `step()`, plus `torch.no_grad()` evaluation

In HW04, you'll build neural networks both from scratch (implementing your own backpropagation) and with PyTorch's tools, train them on EMNIST, and use them to read the ransom note letters. Time to file your report!"""))

# ═══ CELL: Task 06 Header ═══
cells.append(md("""## Task 06: Generate Police Report (0 pts.)

Run the code cell below to generate a report for the Police and submit it on Canvas!

### Task 06: Code (0 pts.)"""))

# ═══ CELL: Task 06 Export Code ═══
cells.append(code("""import os, json

ASS_PATH = "nd-cse-30124-homeworks/labs"
ASS = "lab03"

try:
    from google.colab import _message, files

    repo_ipynb_path = f"/content/{ASS_PATH}/{ASS}/{ASS}.ipynb"

    nb = _message.blocking_request("get_ipynb", timeout_sec=1)["ipynb"]

    os.makedirs(os.path.dirname(repo_ipynb_path), exist_ok=True)
    with open(repo_ipynb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f)

    !jupyter nbconvert --to html "{repo_ipynb_path}"
    files.download(repo_ipynb_path.replace(".ipynb", ".html"))
except:
    import subprocess

    nb_fp = os.getcwd() + f'/{ASS}.ipynb'
    print(os.getcwd())

    subprocess.run(["jupyter", "nbconvert", "--to", "html", nb_fp], check=True)
finally:
    print('[WARNING]: Unable to export notebook as .html')"""))

# ═══ Build notebook ═══
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("lab03_solutions.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Generated notebook with {len(cells)} cells")
