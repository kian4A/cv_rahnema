# Computer Vision & Deep Learning — Rahnema Bootcamp

A practice assignment from **Rahnema College’s Machine Learning Bootcamp**, exploring how neural networks process images and propagate gradients across **depth** and **time**. The notebook implements foundational layers, compares plain and residual CNNs, and investigates memory in recurrent networks.

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

## Run in Google Colab

[Open the notebook in Google Colab](https://colab.research.google.com/github/kian4A/cv_rahnema/blob/main/project_cv.ipynb)

1. Open the notebook using the link above, or upload `project_cv.ipynb` to Google Colab.
2. Choose **Runtime → Change runtime type** and select a **GPU** accelerator (T4 if available).
3. In Part 2, change `download=False` to `download=True` in **both** CIFAR-10 dataset calls. This allows the dataset to download into the fresh Colab runtime:

   ```python
   full_tr = torchvision.datasets.CIFAR10(
       "./data", train=True, download=True, transform=tf_train
   )
   full_te = torchvision.datasets.CIFAR10(
       "./data", train=False, download=True, transform=tf_test
   )
   ```

4. Run the notebook cells from top to bottom. MNIST downloads automatically in Part 3.
5. Save a copy of the completed notebook with its outputs to keep the results and plots.

The notebook uses NumPy, PyTorch, torchvision, Matplotlib, scikit-image, and Pillow. If an import is missing in your Colab runtime, install the corresponding package in a code cell using `%pip install package-name`.

Training time depends on the assigned hardware. Dataset downloads require internet access, and files in the temporary Colab runtime may be lost when the runtime is reset.

## Notes and limitations

- The notebook includes the course assignment text, implementation, explanations, plots, and saved outputs.
- Some written answers contain values from earlier runs, and some assignment prompts refer to 26 layers while the implemented deep model has 50. The tables above use the saved output tables, not those narrative values.
- Gradient measurements are taken on newly initialized models, separately from the training runs.
- Random seeds are set, but exact reproducibility across hardware and software versions is not guaranteed.
- Dataset files, virtual environments, and generated checkpoints should remain outside version control.
