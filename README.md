<div align="center">

![Rose](rose.png)

_**R**ange-**O**f-**S**lice **E**quilibration_
<br>PyTorch Optimizer

Stateless optimization through range-normalized gradient updates.

*In loving memory of my mother, **Rose Kieren**.*

[![PyPI](https://img.shields.io/pypi/v/rose-opt?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/rose-opt/) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![License](https://img.shields.io/badge/license-Apache%202.0-228b22?style=for-the-badge)](LICENSE)<br>
[![Github Sponsors](https://img.shields.io/badge/GitHub%20Sponsors-30363D?&logo=GitHub-Sponsors&logoColor=EA4AAA)](https://github.com/sponsors/MatthewK78) [![PayPal](https://img.shields.io/badge/PayPal-003087?logo=paypal&logoColor=fff)](https://www.paypal.com/donate/?hosted_button_id=VHLPWXMWHJ4C8)

</div>

---

## 📰 News

> 2026-04-26 [v1.0.2] — PyPI support, change to `from rose_opt import Rose`<br>
> 2026-04-19 [v1.0.1] — Misc refinements to algorithm and docs<br>
> 2026-04-17 [v1.0.0] — Initial public release

## 🌹 Introduction

Most adaptive optimizers (such as Adam, RMSprop, and their many variants) accumulate running statistics for every parameter: first-moment estimates, second-moment estimates, and sometimes more. These buffers can double or triple the memory footprint of a model's parameters and introduce temporal entanglements (bias correction, momentum decay, sensitivity to $\beta$ schedules) that make training dynamics harder to reason about.

**Rose** asks a simple question: *how much can you accomplish with just the gradient you have right now?*

At each step, Rose normalizes every gradient tensor by a **per-slice range** yielding one adaptive scale factor per output unit. An optional **coefficient-of-variation trust gate** blends per-slice ranges with their global mean when the ranges are noisy, and optional **gradient centralization** removes shared directional bias before scaling.

## 📦 Installation

```bash
pip install rose-opt
```
or
```bash
pip install git+https://github.com/MatthewK78/Rose
```

Usage:
```python
from rose_opt import Rose

optimizer = Rose(params, lr=1e-3)
```
> **Requires** Python ≥ 3.10 and PyTorch ≥ 2.0

## ✨ Features

| Feature | Detail |
|:--- |:--- |
| **Zero optimizer state** | No momentum, variance estimates, or even step counters. Memory cost is parameters + gradients + working memory, nothing else. |
| **Gradient centralization** | Removes the per-slice mean from gradients of rank ≥ 2, reducing internal covariate shift in the gradient signal and often improving stability and generalization. |
| **CV trust gating** | Automatically detects when per-slice ranges are noisy and gracefully falls back to a robust global estimate. |
| **Decoupled weight decay** | Standard or schedule-coupled weight decay, preventing late-training decay from overpowering vanishing learning rates. |
| **BF16 stochastic rounding** | Unbiased rounding for BFloat16 parameters eliminates systematic truncation drift, meaningfully improving low-precision training fidelity. |
| **Configurable compute precision** | Promotes intermediates to FP64 by default (FP32, BF16, FP16, or native dtype also supported) so that range and division arithmetic stays precise. |

## 🔬 Method

Consider a linear layer with weight matrix $W \in \mathbb{R}^{m \times n}$. Its gradient $G$ has the same shape: $m$ rows, one per output neuron. Rose computes the range ($|\max| - \min$) across the $n$ input-facing elements of each row independently, producing $m$ per-neuron scale factors. Zero-dimensional parameters receive a plain signSGD update.

This is analogous to how Adam assigns each *scalar* parameter its own adaptive denominator via a running variance estimate. Rose instead assigns each *output slice* a denominator based on the instantaneous spread of its gradient, requiring no history at all.

The **trust gate** addresses a practical concern: when per-slice ranges vary wildly (high coefficient of variation), the individual ranges may become unreliable. The trust factor $\tau = \mu / (\mu + \sigma)$ is close to 1 when ranges are self-consistent and close to 0 when they are noisy. The denominator smoothly interpolates between the local range (full detail) and the global mean range (maximum noise resistance).

**Why range instead of variance?** Range is cheaper to compute and maps cleanly to the idea of *scale equilibration*: it answers "how wide is this gradient slice?" rather than "how energetic is it?" In practice, for the shapes common in deep learning, the two carry similar information, but range has the advantage of depending only on two order statistics and producing a scale factor that directly normalizes the gradient's dynamic range.

## 🎛️ Hyperparameters

### `lr`: Learning Rate

The global step size. Because this optimizer uses range-based normalization rather than Adam's RMS-based normalization, the same `lr` value can correspond to very different effective update sizes. Tune `lr` independently rather than relying on Adam defaults.

```python
Rose(params, lr=1e-3)
```

---

### `weight_decay`: Decoupled Weight Decay

| | |
|:--- |:--- |
| **Default** | `1e-4` |
| **Disable** | `0` or `None` |

A decoupled multiplicative coefficient applied independently of the gradient step, shrinking weights toward zero each step. This is the same formulation used by AdamW.

```python
Rose(params, lr=1e-3, weight_decay=1e-4)  # default
Rose(params, lr=1e-3, weight_decay=0)     # disabled
```

---

### `wd_schedule`: Schedule-Coupled Weight Decay

| | |
|:--- |:--- |
| **Default** | `False` |

Scales weight decay proportionally with a learning-rate schedule so that decay weakens as the learning rate drops. This prevents weight decay from dominating the update in the late phase of training when the learning rate is small. The per-step multiplicative factor becomes:

$1 - \frac{\eta_t}{\eta_{\text{ref}}} \cdot \lambda$

| Value | Behavior |
|:--- |:--- |
| `False` | Standard decoupled weight decay.                                                                   |
| `True` | $\eta_{\text{ref}}$ is resolved from `max_lr` → `initial_lr` → constructor `lr`. |
| `float` | The provided value is used directly as $\eta_{\text{ref}}$. |

```python
Rose(params, lr=1e-3, weight_decay=1e-4, wd_schedule=True)  # auto reference
Rose(params, lr=1e-3, weight_decay=1e-4, wd_schedule=1e-3)  # explicit reference
```

---

### `centralize`: Gradient Centralization

| | |
|:--- |:--- |
| **Default** | `True` |

Subtracts the mean of each gradient slice along the non-leading axes before the range computation. Only applies to parameters with rank ≥ 2, biases and other 1-D parameters are never centralized.

Gradient centralization constrains updates in the subspace orthogonal to the slice mean, which can act as a mild regularizer and improve training stability.

```python
Rose(params, lr=1e-3, centralize=True)   # default
Rose(params, lr=1e-3, centralize=False)  # disabled
```

---

### `stabilize`: Coefficient-of-Variation Trust Gating

| | |
|:--- |:--- |
| **Default** | `True` |

Computes a trust factor from the coefficient of variation of the per-slice range tensor and interpolates between the local per-slice range and the global mean range. This can smooth noisy range estimates. Some models perform better with it enabled, others disabled; try both.

- **Trust ≈ 1** (consistent ranges) → local detail preserved.
- **Trust ≈ 0** (noisy ranges) → smooth global fallback.

```python
Rose(params, lr=1e-3, stabilize=True)   # default
Rose(params, lr=1e-3, stabilize=False)  # raw per-slice ranges only
```

---

### `bf16_sr`: BFloat16 Stochastic Rounding

| | |
|:--- |:--- |
| **Default** | `True` |

When a parameter is stored in BFloat16, promotes it to higher precision for the update, then stochastically rounds on write-back. This produces statistically unbiased rounding, correcting for the systematic truncation drift that BF16's limited mantissa otherwise introduces. Has no effect on parameters with any other dtype.

| Value | Effect |
|:--- |:--- |
| `False` | BF16 stochastic rounding is disabled. |
| `True` | Uses the default random-number generator. |
| `torch.Generator` | Treats `bf16_sr` as enabled and forwards the generator to `random_`. Useful for deterministic output. |

```python
Rose(params, lr=1e-3, bf16_sr=True)   # default
Rose(params, lr=1e-3, bf16_sr=False)  # plain truncation

sr_gen = torch.Generator(device="cuda").manual_seed(0xd1ce)
Rose(params, lr=1e-3, bf16_sr=sr_gen)
```

---

### `compute_dtype`: Internal Compute Precision

| | |
|:--- |:--- |
| **Default** | `"fp64"` |

The dtype to which parameters and gradients are promoted for the update step. FP64 is recommended; the intermediate range computation and division benefit from the extra mantissa bits, especially for parameters with large fan-in or near-zero gradient spread.

| Value | Effect  |
|:--- |:--- |
| `torch.float64` / `"fp64"` | Full FP64 precision for all intermediates. |
| `torch.float32` / `"fp32"` | Reasonable fallback when FP64 is too costly. |
| `torch.float16` / `"fp16"` | Not generally recommended; listed for completeness. |
| `torch.bfloat16` / `"bf16"` | Not generally recommended; listed for completeness. |
| `None` / `"none"` | No promotion; compute in native dtype _(however, BF16 params still use FP32 if `bf16_sr=True`)._ |

```python
Rose(params, lr=1e-3, compute_dtype="fp64")         # default
Rose(params, lr=1e-3, compute_dtype=torch.float32)  # lighter
Rose(params, lr=1e-3, compute_dtype=None)           # native
```

## 💖 Acknowledgements

This optimizer is named in loving memory of my mother, Rose Kieren, who always listened intently with genuine interest as I rambled about AI and computers, and whose unconditional love and presence through both the best and hardest of times made all of this possible.

I am deeply grateful to my late father for always taking an interest in my exploration of technology, and for the encouragement and warmth he brought to every conversation.

To my wife, for her extraordinary patience through the countless days and nights I've spent absorbed in programming and research, and for her unwavering intellectual and emotional support.

To my son, a source of joy, perspective, and inspiration, whose curiosity and bright spirit remind me why building for the future matters.

As an independent researcher, I am grateful for the love and support of all of my family and friends. This work would not have been possible without them.

## 😊 A Kind Request

If you use Rose in your research, project, or product, I would be grateful if you would **mention it by name** and credit its author, **Matthew E. Kieren**. A citation (see below), a footnote, a line in your README; any acknowledgment, however small, helps motivate me to do more. And if you have a moment, I would love to hear your story.

If you'd like to support my ongoing development efforts of this project and others, you can send a donation here on [**GitHub**](https://github.com/sponsors/MatthewK78), or through [**PayPal**](https://www.paypal.com/donate/?hosted_button_id=VHLPWXMWHJ4C8).

Your support and acknowledgment are sincerely appreciated! 😊

## 📄 Citation

```bibtex
@software{kieren2026rose,
  author       = {Kieren, Matthew E.},
  title        = {Rose: Range-Of-Slice Equilibration optimizer},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19589764},
  url          = {https://doi.org/10.5281/zenodo.19589764}
}
```

## 📚 References

<sup>1</sup> Kingma, D. P. & Ba, J. (2014), *Adam: A Method for Stochastic Optimization.* arXiv:[1412.6980](https://arxiv.org/abs/1412.6980)

<sup>2</sup> Loshchilov, I. & Hutter, F. (2017), *Decoupled Weight Decay Regularization.* arXiv:[1711.05101](https://arxiv.org/abs/1711.05101)

<sup>3</sup> Hazan, E., Levy, K. Y., & Shalev-Shwartz, S. (2015), *Beyond Convexity: Stochastic Quasi-Convex Optimization*. arXiv:[1507.02030](https://arxiv.org/abs/1507.02030)

<sup>4</sup> You, Y., Gitman, I., & Ginsburg, B. (2017), *Large Batch Training of Convolutional Networks*. arXiv:[1708.03888](https://arxiv.org/abs/1708.03888)

<sup>5</sup> Yu, A. W., Huang, L., Lin, Q., Salakhutdinov, R., & Carbonell, J. (2017), *Block-Normalized Gradient Method: An Empirical Study for Training Deep Neural Network*. arXiv:[1707.04822](https://arxiv.org/abs/1707.04822)

<sup>6</sup> Bernstein, J., Wang, Y. X., Azizzadenesheli, K., & Anandkumar, A. (2018), *signSGD: Compressed Optimisation for Non-Convex Problems*. arXiv:[1802.04434](https://arxiv.org/abs/1802.04434)

<sup>7</sup> Yong, H., Huang, J., Hua, X. & Zhang, L. (2020). *Gradient Centralization: A New Optimization Technique for Deep Neural Networks.* arXiv:[2004.01461](https://arxiv.org/abs/2004.01461)

<sup>8</sup> Zamirai, P., Zhang, J., Aberger, C. R. & De Sa, C. (2020). *Revisiting BFloat16 Training.* arXiv:[2010.06192](https://arxiv.org/abs/2010.06192)

## ⚖️ License

Copyright <sup>©</sup> 2026 Matthew Everet Kieren

Licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). You may use, modify, and distribute this software in accordance with the license. See [`LICENSE`](LICENSE) for the full text.

---

<div align="center">

https://github.com/MatthewK78/Rose

</div>