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
| 01       | PyTorch Tensors                                  | 1      |
| &nbsp;&nbsp;&nbsp;&nbsp;01-1     | &nbsp;&nbsp;&nbsp;&nbsp;- Creating and inspecting tensors  | 0      |
| &nbsp;&nbsp;&nbsp;&nbsp;01-2     | &nbsp;&nbsp;&nbsp;&nbsp;- NumPy ↔ Tensor conversion       | 1      |
| 02       | Tensor Operations                                | 1      |
| &nbsp;&nbsp;&nbsp;&nbsp;02-1     | &nbsp;&nbsp;&nbsp;&nbsp;- Reshaping and permuting demo     | 0      |
| &nbsp;&nbsp;&nbsp;&nbsp;02-2     | &nbsp;&nbsp;&nbsp;&nbsp;- Prepare images for a CNN         | 1      |
| 03       | Building a Model with nn.Module                  | 1      |
| &nbsp;&nbsp;&nbsp;&nbsp;03-1     | &nbsp;&nbsp;&nbsp;&nbsp;- nn.Module pattern demo           | 0      |
| &nbsp;&nbsp;&nbsp;&nbsp;03-2     | &nbsp;&nbsp;&nbsp;&nbsp;- Define a small classifier        | 1      |
| 04       | Dataset and DataLoader                           | 1      |
| &nbsp;&nbsp;&nbsp;&nbsp;04-1     | &nbsp;&nbsp;&nbsp;&nbsp;- Data loading demo                | 0      |
| &nbsp;&nbsp;&nbsp;&nbsp;04-2     | &nbsp;&nbsp;&nbsp;&nbsp;- Create a DataLoader              | 1      |
| 05       | The Training Loop                                | 1      |
| &nbsp;&nbsp;&nbsp;&nbsp;05-1     | &nbsp;&nbsp;&nbsp;&nbsp;- Training loop demo               | 0      |
| &nbsp;&nbsp;&nbsp;&nbsp;05-2     | &nbsp;&nbsp;&nbsp;&nbsp;- Train and evaluate               | 1      |
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
# TASK 01: PyTorch Tensors
# ═══════════════════════════════════════════════════════════════
cells.append(md("""## Task 01: PyTorch Tensors (1 pt.)
### Task 01-1: Description (0 pts.)

PyTorch's fundamental data structure is the **tensor** — a multi-dimensional array, just like NumPy's `ndarray`. In fact, they're so similar that most operations have the same name. The key difference: PyTorch tensors can run on GPUs and track gradients for automatic differentiation.

---

#### Creating Tensors

[`torch.tensor()`](https://pytorch.org/docs/stable/generated/torch.tensor.html) creates a tensor from data (like a list or NumPy array):

```python
# From a Python list
x = torch.tensor([1, 2, 3])           # shape: (3,)
y = torch.tensor([[1, 2], [3, 4]])     # shape: (2, 2)

# Common constructors
torch.zeros(3, 4)      # 3×4 matrix of zeros
torch.ones(2, 3)       # 2×3 matrix of ones
torch.randn(5, 5)      # 5×5 matrix of random normal values
torch.arange(10)        # [0, 1, 2, ..., 9]
```

---

#### Inspecting Tensors

Every tensor has a shape, data type, and device:

```python
x = torch.randn(3, 4)
x.shape       # torch.Size([3, 4])
x.dtype       # torch.float32 (default)
x.device      # device(type='cpu')
```

---

#### NumPy vs PyTorch — Quick Reference

| NumPy | PyTorch | Notes |
|-------|---------|-------|
| `np.array([1,2,3])` | `torch.tensor([1,2,3])` | Create from list |
| `np.zeros((3,4))` | `torch.zeros(3,4)` | Note: no tuple in PyTorch |
| `np.random.randn(3,4)` | `torch.randn(3,4)` | Random normal |
| `arr.shape` | `tensor.shape` | Identical |
| `arr.dtype` | `tensor.dtype` | `float64` vs `float32` default |
| `arr.reshape(2,3)` | `tensor.reshape(2,3)` | Identical syntax |
| `arr.T` | `tensor.T` | Transpose |
| `arr @ arr2` | `tensor @ tensor2` | Matrix multiply |

---

#### Converting Between NumPy and PyTorch

[`torch.from_numpy()`](https://pytorch.org/docs/stable/generated/torch.from_numpy.html) and [`.numpy()`](https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html) convert between formats:

```python
# NumPy -> Tensor
arr = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(arr)          # shares memory!
tensor = torch.tensor(arr)              # copies data (safer)

# Tensor -> NumPy
arr_back = tensor.numpy()               # shares memory (CPU only)
arr_back = tensor.detach().cpu().numpy() # safe version (works from GPU too)
```

**Important:** `torch.from_numpy()` shares memory with the original array — changes to one affect the other. Use `torch.tensor()` if you want an independent copy.

---

#### Moving Tensors Between Devices

[`.to(device)`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html) moves a tensor to a specific device:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.randn(3, 4)
x = x.to(device)         # move to GPU (or stay on CPU)
print(x.device)           # device(type='cuda', index=0)
```

### Task 01-1: Code (0 pts.)"""))

# ═══ CELL: Task 01-1 Demo Code ═══
cells.append(code("""# Creating tensors
a = torch.tensor([1, 2, 3])
b = torch.zeros(3, 4)
c = torch.randn(2, 3)
d = torch.arange(12)

print(f"a: shape={a.shape}, dtype={a.dtype}, values={a}")
print(f"b: shape={b.shape}, dtype={b.dtype}")
print(f"c: shape={c.shape}, dtype={c.dtype}")
print(f"d: shape={d.shape}, values={d}")

# NumPy <-> Tensor conversion
arr = np.array([[1.0, 2.0], [3.0, 4.0]])
tensor_from_np = torch.from_numpy(arr)       # shares memory
tensor_copy = torch.tensor(arr)               # independent copy

print(f"\\nNumPy array dtype:  {arr.dtype}")
print(f"from_numpy dtype:   {tensor_from_np.dtype}")   # float64 (matches NumPy)
print(f"torch.tensor dtype: {tensor_copy.dtype}")       # float64 (matches input)

# Shared memory demo
arr[0, 0] = 999
print(f"\\nAfter modifying NumPy array:")
print(f"  from_numpy sees change: {tensor_from_np[0, 0]}")  # 999
print(f"  torch.tensor is independent: {tensor_copy[0, 0]}")  # 1.0

# Device
print(f"\\nDevice: {tensor_copy.device}")"""))

# ═══ CELL: Task 01-2 Exercise Description ═══
cells.append(md("""### Task 01-2: NumPy ↔ Tensor Conversion (1 pt.)

The EMNIST dataset used by the forensics classifier is stored as NumPy `.npz` files. Load the training data, convert the images and labels to PyTorch tensors, and verify the conversion.

Steps:
1. Load the `.npz` file with `np.load()`
2. Extract the `X` (images) and `y` (labels) arrays
3. Flatten images from `(n, 28, 28)` to `(n, 784)` and normalize pixel values to `[0, 1]`
4. Convert images to a `float32` tensor and labels to a `long` tensor (required by PyTorch's loss functions)
5. Print shapes, dtypes, and pixel value range

### Task 01-2: Code (1 pt.)"""))

# ═══ CELL: Task 01-2 Code ═══
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

# ═══ CELL: Task 01-2 Expected Output ═══
cells.append(md("""### Task 01-2: Expected Output (1 pt.)
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
# TASK 02: Tensor Operations
# ═══════════════════════════════════════════════════════════════
cells.append(md("""## Task 02: Tensor Operations (1 pt.)
### Task 02-1: Description (0 pts.)

Neural networks are picky about the shape of their input data. **Fully connected networks** (FFNs) expect flat vectors: `(batch_size, n_features)`. **Convolutional networks** (CNNs) expect image tensors: `(batch_size, channels, height, width)`. You'll constantly need to reshape between these formats.

---

#### Reshaping Tensors

[`tensor.reshape()`](https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html) and [`tensor.view()`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html) change shape without changing data (same as NumPy):

```python
x = torch.arange(12)          # shape: (12,)
x.reshape(3, 4)                # shape: (3, 4)
x.reshape(-1, 4)               # shape: (3, 4)  — -1 = auto
x.reshape(2, 2, 3)             # shape: (2, 2, 3)
```

[`tensor.view()`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html) is identical to `reshape` but requires contiguous memory. Prefer `reshape` — it always works.

---

#### Permuting Dimensions

[`tensor.permute()`](https://pytorch.org/docs/stable/generated/torch.Tensor.permute.html) reorders dimensions — essential for converting between image formats:

```python
# Image loaded as (H, W, C) — e.g., from NumPy/OpenCV
img = torch.randn(28, 28, 3)     # (H, W, C)

# PyTorch CNNs expect (C, H, W)
img_chw = img.permute(2, 0, 1)   # (3, 28, 28)
```

The numbers in `permute()` are the **old dimension indices** in the **new order**. `permute(2, 0, 1)` means: "put old dim 2 first, then old dim 0, then old dim 1."

---

#### Adding and Removing Dimensions

[`tensor.unsqueeze(dim)`](https://pytorch.org/docs/stable/generated/torch.Tensor.unsqueeze.html) adds a size-1 dimension; [`tensor.squeeze()`](https://pytorch.org/docs/stable/generated/torch.Tensor.squeeze.html) removes them:

```python
x = torch.randn(28, 28)         # shape: (28, 28)

# Add batch dimension at position 0
x.unsqueeze(0)                   # shape: (1, 28, 28)

# Add channel dimension at position 0, then batch at position 0
x.unsqueeze(0).unsqueeze(0)      # shape: (1, 1, 28, 28)

# Remove all size-1 dimensions
y = torch.randn(1, 1, 28, 28)
y.squeeze()                      # shape: (28, 28)
```

---

#### Flattening

[`tensor.flatten(start_dim)`](https://pytorch.org/docs/stable/generated/torch.Tensor.flatten.html) collapses dimensions from `start_dim` onward:

```python
x = torch.randn(32, 64, 4, 4)    # (batch, channels, h, w) — CNN output
x.flatten(1)                      # (32, 1024) — flatten everything after batch dim
```

This is how CNN feature maps are converted to vectors for fully connected layers.

---

#### The Shape Pipeline: Flat → Image → CNN-ready

```
Flat vector:    (784,)           — stored in dataset
     ↓ reshape(28, 28)
2D image:       (28, 28)         — for visualization
     ↓ unsqueeze(0)
With channel:   (1, 28, 28)      — single grayscale channel
     ↓ unsqueeze(0)
Batched:        (1, 1, 28, 28)   — ready for CNN forward pass
```

### Task 02-1: Code (0 pts.)"""))

# ═══ CELL: Task 02-1 Demo Code ═══
cells.append(code("""# Reshape: flat -> 2D image -> display
flat_img = images_tensor[0]              # shape: (784,)
img_2d = flat_img.reshape(28, 28)         # shape: (28, 28)
print(f"Flat: {flat_img.shape} -> 2D: {img_2d.shape}")

# Permute example: (H, W, C) -> (C, H, W)
fake_color = torch.randn(28, 28, 3)      # like a NumPy/cv2 image
chw = fake_color.permute(2, 0, 1)
print(f"\\n(H,W,C): {fake_color.shape} -> (C,H,W): {chw.shape}")

# Unsqueeze: add batch and channel dims
img_with_channel = img_2d.unsqueeze(0)           # (1, 28, 28) — add channel
img_batched = img_with_channel.unsqueeze(0)       # (1, 1, 28, 28) — add batch
print(f"\\n2D: {img_2d.shape} -> +channel: {img_with_channel.shape} -> +batch: {img_batched.shape}")

# Flatten: collapse spatial dims for FC layer
batch = torch.randn(32, 64, 4, 4)    # fake CNN output
flat = batch.flatten(1)               # flatten everything after batch dim
print(f"\\nCNN output: {batch.shape} -> flattened: {flat.shape}")

# Display the first image
plt.figure(figsize=(3, 3))
plt.imshow(img_2d.numpy(), cmap='gray')
plt.title(f"Label: {labels_tensor[0].item()}")
plt.axis('off')
plt.show()"""))

# ═══ CELL: Task 02-2 Exercise ═══
cells.append(md("""### Task 02-2: Prepare Images for a CNN (1 pt.)

The forensics CNN expects input shaped `(batch_size, 1, 28, 28)` — batch of single-channel 28×28 images. Your data is currently `(n_samples, 784)` flat vectors. Reshape the entire dataset.

Steps:
1. Reshape `images_tensor` from `(n_samples, 784)` to `(n_samples, 28, 28)` using `.reshape()`
2. Add a channel dimension to get `(n_samples, 1, 28, 28)` using `.unsqueeze()`
3. Print the shape at each step
4. Verify by displaying a grid of the first 8 images

### Task 02-2: Code (1 pt.)"""))

# ═══ CELL: Task 02-2 Code ═══
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

# ═══ CELL: Task 02-2 Expected Output ═══
cells.append(md("""### Task 02-2: Expected Output (1 pt.)
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
# TASK 03: nn.Module
# ═══════════════════════════════════════════════════════════════
cells.append(md("""## Task 03: Building a Model with nn.Module (1 pt.)
### Task 03-1: Description (0 pts.)

Every PyTorch model is a Python class that inherits from [`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). This gives your model automatic parameter tracking, GPU support, and integration with PyTorch's training tools.

---

#### The nn.Module Pattern

Every model follows the same two-method pattern:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()                    # MUST call this first
        self.layer1 = nn.Linear(784, 128)     # define layers here
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):                     # define computation here
        x = F.relu(self.layer1(x))            # layer1 + activation
        x = self.layer2(x)                    # output layer (no activation)
        return x
```

**`__init__`**: Define all layers (learnable parameters) as `self.something`. Always call `super().__init__()` first.

**`forward`**: Define how data flows through the layers. This is called when you do `model(x)`.

---

#### Key Layer Types

[`nn.Linear(in_features, out_features)`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) — a fully connected layer:

```python
fc = nn.Linear(784, 128)    # 784 inputs -> 128 outputs
# Contains: weight matrix (128, 784) + bias vector (128,)

output = fc(input)           # input shape: (batch, 784) -> output: (batch, 128)
```

---

#### Activation Functions

[`F.relu()`](https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html) — Rectified Linear Unit, the most common activation:

```python
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
F.relu(x)    # tensor([0., 0., 0., 1., 2.])  — zeros out negatives
```

---

#### Weight Initialization

[`nn.init.kaiming_normal_()`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_) (He initialization) is the standard for ReLU networks:

```python
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```

---

#### Using the Model

```python
model = MyModel()                       # create instance
print(model)                            # see architecture summary

output = model(input_tensor)            # forward pass (calls .forward())
print(output.shape)                     # check output shape

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
```

### Task 03-1: Code (0 pts.)"""))

# ═══ CELL: Task 03-1 Demo Code ═══
cells.append(code("""# A minimal 1-layer model
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 47)    # 784 inputs -> 47 outputs

    def forward(self, x):
        return self.fc(x)

model = TinyModel()
print(model)
print(f"\\nParameters: {sum(p.numel() for p in model.parameters())}")

# Forward pass with a single fake sample
fake_input = torch.randn(1, 784)        # (batch=1, features=784)
output = model(fake_input)
print(f"\\nInput shape:  {fake_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output (raw logits): {output.data}")"""))

# ═══ CELL: Task 03-2 Exercise ═══
cells.append(md("""### Task 03-2: Define a Small Classifier (1 pt.)

Build a simple 3-layer fully connected classifier for the 47-class EMNIST dataset:

- Input: 784 features (28×28 flattened image)
- Hidden layer 1: 256 units + ReLU activation
- Hidden layer 2: 128 units + ReLU activation
- Output layer: 47 units (one per class, no activation)

Steps:
1. Define the class inheriting from `nn.Module`
2. Create three `nn.Linear` layers in `__init__`
3. Apply `F.relu()` after the first two layers in `forward()`
4. Create an instance and run a forward pass on a small batch to verify

### Task 03-2: Code (1 pt.)"""))

# ═══ CELL: Task 03-2 Code ═══
cells.append(code("""# TODO: Define the classifier
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 47)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

# ═══ CELL: Task 03-2 Expected Output ═══
cells.append(md("""### Task 03-2: Expected Output (1 pt.)
```
SimpleClassifier(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=47, bias=True)
)

Total parameters: 239,535

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
# TASK 04: Dataset and DataLoader
# ═══════════════════════════════════════════════════════════════
cells.append(md("""## Task 04: Dataset and DataLoader (1 pt.)
### Task 04-1: Description (0 pts.)

Neural networks are trained on **mini-batches** — small chunks of data processed at a time. PyTorch's `DataLoader` handles batching, shuffling, and iteration automatically. But first, you need to wrap your data in a `Dataset`.

---

#### TensorDataset — The Quick Way

[`TensorDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset) wraps tensors into a dataset. Each sample is a tuple of corresponding elements:

```python
from torch.utils.data import TensorDataset

X = torch.randn(1000, 784)     # 1000 images
y = torch.randint(0, 47, (1000,))  # 1000 labels

dataset = TensorDataset(X, y)
print(len(dataset))              # 1000
sample_x, sample_y = dataset[0]  # get first sample
```

---

#### Custom Dataset — For More Control

For complex loading (e.g., reading images from disk), subclass [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset):

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
```

You **must** implement `__len__()` and `__getitem__()`.

---

#### DataLoader — Batching and Shuffling

[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) wraps a dataset and produces batches:

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over batches
for batch_x, batch_y in loader:
    print(batch_x.shape)  # (32, 784) — or smaller for last batch
    print(batch_y.shape)  # (32,)
    break  # just show first batch
```

| Parameter | What it does |
|-----------|-------------|
| `batch_size` | Number of samples per batch |
| `shuffle` | Randomize order each epoch (True for training, False for testing) |

---

#### Train / Validation / Test Splits

[`random_split()`](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split) divides a dataset into non-overlapping subsets:

```python
from torch.utils.data import random_split

train_set, val_set = random_split(dataset, [800, 200])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
```

### Task 04-1: Code (0 pts.)"""))

# ═══ CELL: Task 04-1 Demo Code ═══
cells.append(code("""# Quick demo with TensorDataset
small_X = images_tensor[:100]
small_y = labels_tensor[:100]

dataset = TensorDataset(small_X, small_y)
print(f"Dataset size: {len(dataset)}")

# Access one sample
sample_x, sample_y = dataset[0]
print(f"Sample: x shape={sample_x.shape}, y={sample_y}")

# DataLoader
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Iterate one batch
for batch_x, batch_y in loader:
    print(f"\\nBatch: x shape={batch_x.shape}, y shape={batch_y.shape}")
    print(f"Labels in this batch: {batch_y}")
    break"""))

# ═══ CELL: Task 04-2 Exercise ═══
cells.append(md("""### Task 04-2: Create a DataLoader (1 pt.)

Create a proper training pipeline from the full EMNIST dataset:

1. Create a `TensorDataset` from `images_tensor` and `labels_tensor`
2. Split into training (80%) and validation (20%) sets using `random_split`
3. Create `DataLoader`s for each — training with `shuffle=True`, validation with `shuffle=False`
4. Iterate over one training batch and print its shapes

### Task 04-2: Code (1 pt.)"""))

# ═══ CELL: Task 04-2 Code ═══
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

# ═══ CELL: Task 04-2 Expected Output ═══
cells.append(md("""### Task 04-2: Expected Output (1 pt.)
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
# TASK 05: Training Loop
# ═══════════════════════════════════════════════════════════════
cells.append(md("""## Task 05: The Training Loop (1 pt.)
### Task 05-1: Description (0 pts.)

Training a neural network boils down to repeating three steps on each batch of data. This is the most important pattern in PyTorch.

---

#### The 3-Step Training Loop

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        # 1. Forward pass — compute predictions and loss
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        # 2. Backward pass — compute gradients
        optimizer.zero_grad()    # clear old gradients
        loss.backward()          # compute new gradients

        # 3. Update — adjust weights
        optimizer.step()         # apply gradients to weights
```

**That's it.** Every neural network in PyTorch is trained with these same three lines inside the batch loop.

---

#### The Optimizer

[`torch.optim.Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) is the most common optimizer. It automatically adjusts the learning rate per-parameter:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

`model.parameters()` gives the optimizer access to all learnable weights. `lr` is the learning rate (how big each update step is).

---

#### The Loss Function

[`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) is the standard loss for multi-class classification. It expects:
- **Input**: raw logits of shape `(batch_size, n_classes)` — do NOT apply softmax first
- **Target**: class indices of shape `(batch_size,)` with dtype `long`

```python
criterion = nn.CrossEntropyLoss()

logits = torch.randn(4, 47)              # 4 samples, 47 classes
targets = torch.tensor([0, 15, 32, 7])   # correct class for each
loss = criterion(logits, targets)         # single scalar value
```

---

#### Evaluation Mode

When evaluating (not training), disable gradient computation for speed and use `model.eval()`:

```python
model.eval()                          # turn off dropout, batchnorm, etc.
with torch.no_grad():                 # disable gradient tracking
    logits = model(val_batch_x)
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == val_batch_y).float().mean()
```

[`torch.no_grad()`](https://pytorch.org/docs/stable/generated/torch.no_grad.html) saves memory and computation. Always use it during evaluation.

[`torch.argmax(logits, dim=1)`](https://pytorch.org/docs/stable/generated/torch.argmax.html) returns the index of the largest value along dimension 1 — i.e., the predicted class.

Don't forget to call `model.train()` before the next training epoch!

### Task 05-1: Code (0 pts.)"""))

# ═══ CELL: Task 05-1 Demo Code ═══
cells.append(code("""# Demo: train a tiny model for 3 epochs on a small subset
demo_dataset = TensorDataset(images_tensor[:500], labels_tensor[:500])
demo_loader = DataLoader(demo_dataset, batch_size=32, shuffle=True)

demo_model = TinyModel()    # from Task 03 — just 1 layer
optimizer = torch.optim.Adam(demo_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    demo_model.train()
    total_loss = 0
    for batch_x, batch_y in demo_loader:
        logits = demo_model(batch_x)          # 1. Forward
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()                  # 2. Backward
        loss.backward()

        optimizer.step()                       # 3. Update

        total_loss += loss.item()

    avg_loss = total_loss / len(demo_loader)
    print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")"""))

# ═══ CELL: Task 05-2 Exercise ═══
cells.append(md("""### Task 05-2: Train and Evaluate (1 pt.)

Train the `SimpleClassifier` from Task 03 on the training data and evaluate on the validation data.

Steps:
1. Create an `Adam` optimizer with `lr=0.001` and a `CrossEntropyLoss` criterion
2. Train for **5 epochs**, printing the average loss each epoch
3. After training, evaluate on the validation set: compute accuracy using `torch.argmax` and `torch.no_grad()`

### Task 05-2: Code (1 pt.)"""))

# ═══ CELL: Task 05-2 Code ═══
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

# ═══ CELL: Task 05-2 Expected Output ═══
cells.append(md("""### Task 05-2: Expected Output (1 pt.)
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
