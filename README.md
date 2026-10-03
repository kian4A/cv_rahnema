# Computer Vision & Deep Learning — Rahnema Bootcamp

An educational project exploring how neural networks process images and propagate gradients across **depth** and **time**. The notebook implements foundational layers, compares plain and residual CNNs, and investigates memory in recurrent networks.

**Author:** Kian Aghmashe  
**Context:** Introduction to Machine Learning — Rahnema College ML Bootcamp  
**Notebook:** [project_cv.ipynb](project_cv.ipynb)

## Project overview

### 1. Convolution and pooling from scratch

- NumPy implementations of 2D convolution and max pooling, including forward and backward passes.
- Numerical gradient checks and assertions for correctness.
- Handcrafted grayscale, Sobel-x, Sobel-y, and box-blur filters applied to sample images.

### 2. Network depth and residual connections

- Configurable PyTorch CNNs with 8 or 50 weight layers, with and without residual connections.
- Projection shortcuts when spatial resolution and channel counts change.
- Training on a fixed 20,000-image CIFAR-10 subset and evaluation on the 10,000-image test set.
- Comparison of training loss, test accuracy, and per-element RMS gradients across blocks.
- Ten training epochs per configuration, using SGD and cosine learning-rate scheduling.

These are custom educational architectures, not torchvision's standard ResNet models.

### 3. Sequence memory: RNN vs. LSTM

- A CNN encoder shared across MNIST frames.
- Manually implemented RNN and LSTM recurrences using PyTorch operations and automatic differentiation, without `nn.RNN` or `nn.LSTM`.
- Sequences of 4 or 24 frames, with the target defined as `(first_digit + last_digit) % 10`.
- Six training epochs per configuration and gradient analysis across timesteps.

## Recorded results

The following values come from the uploaded notebook's saved output tables. They have not been independently reproduced and may vary across environments.

### CIFAR-10 depth experiment

| Model | Final training loss | Test accuracy |
| --- | ---: | ---: |
| Plain CNN — 8 layers | 0.6545 | 74.63% |
| Residual CNN — 8 layers | 0.6476 | 74.85% |
| Plain CNN — 50 layers | 1.7136 | 37.22% |
| Residual CNN — 50 layers | 0.9454 | 65.98% |

Residual connections improve optimization of the deeper model in this run, but the 8-layer models still achieve higher test accuracy under the given training budget. The initial gradient spread across blocks is approximately `1.02e3` for the deep plain model and `2.08e1` for the deep residual model.

### MNIST sequence experiment

| Model | Sequence length | Final training loss | Test accuracy |
| --- | ---: | ---: | ---: |
| RNN | 4 | 2.3004 | 10.30% |
| LSTM | 4 | 2.3026 | 9.25% |
| RNN | 24 | 2.3029 | 9.95% |
| LSTM | 24 | 2.3030 | 9.20% |

All four sequence runs remain close to the 10% chance baseline. The saved experiments therefore do **not** demonstrate successful learning of the sequence task or an accuracy improvement from using an LSTM. Gradient-flow diagnostics and task accuracy should be interpreted separately; further investigation is needed.

## Setup and execution

Use a Python environment with NumPy, PyTorch, torchvision, Matplotlib, scikit-image, Pillow, and JupyterLab. A CUDA-capable GPU is recommended for the training sections; the notebook falls back to CPU when CUDA is unavailable.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy torch torchvision matplotlib scikit-image pillow jupyterlab
```

For GPU execution, use a PyTorch installation compatible with your machine's CUDA setup. Exact dependency versions are not pinned in this repository.

### Prepare CIFAR-10

The notebook currently loads CIFAR-10 with `download=False`. Before running it for the first time, execute this from the repository directory:

```bash
python - <<'PY'
from torchvision.datasets import CIFAR10
CIFAR10(root='./data', train=True, download=True)
CIFAR10(root='./data', train=False, download=True)
PY
```

MNIST is downloaded automatically by the notebook. Sample photos come from `skimage.data`; the notebook's fallback image paths require local files that are not included.

### Open the notebook

```bash
jupyter lab project_cv.ipynb
```

Select the project environment and run cells from top to bottom. Initial dataset downloads require internet access. For Google Colab, enable a GPU runtime and prepare CIFAR-10 before its loading cell.

## Notes and limitations

- The notebook includes the course assignment text, implementation, explanations, plots, and saved outputs.
- Some written answers contain values from earlier runs, and some assignment prompts refer to 26 layers while the implemented deep model has 50. The tables above use the saved output tables, not those narrative values.
- Gradient measurements are taken on newly initialized models, separately from the training runs.
- Random seeds are set, but exact reproducibility across hardware and software versions is not guaranteed.
- Dataset files, virtual environments, and generated checkpoints should remain outside version control.
