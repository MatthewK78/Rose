# Copyright 2026 Matthew Everet Kieren. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

class Rose(torch.optim.Optimizer):
    """Rose: Range-Of-Slice Equilibration optimizer.
    In loving memory of my mother, Rose Kieren.
    
    Copyright 2026 Matthew Everet Kieren. All Rights Reserved.
    Licensed under the Apache License, Version 2.0.
    
    Rose rescales gradients using a per-slice `max - min` range
    computed by reducing all dimensions beyond the leading axis.
    Unlike Adam and other stateful optimizers, Rose maintains no
    per-parameter state between steps: no momentum buffers,
    variance estimates, or step counters.
    
    Args:
        params (iterable):
            Iterable of model parameters or parameter-group dictionaries.
        
        lr (float):
            --- Learning Rate ---
            
            Start with values you would typically try for Adam (e.g.,
            `1e-3`). However, because the denominator is range-based
            rather than RMS-based, effective step magnitudes differ from
            those of Adam.
        
        weight_decay (float or None, optional) [1e-4]:
            --- Decoupled Weight Decay ---
            
            A decoupled multiplicative weight-decay coefficient applied
            separately from the adaptive gradient step. It gently
            shrinks weights toward zero at each step and can help reduce
            overfitting. Set to `0` or `None` to disable it.
        
        wd_schedule (bool or float, optional) [False]:
            --- Schedule-Coupled Weight Decay ---
            
            Scales weight decay proportionally with the learning-rate
            schedule so that decay weakens as the learning rate drops,
            preventing it from overpowering small updates. The per-step
            factor becomes `1 - (lr / lr_ref) * weight_decay`.
            
            If `False`, standard decoupled weight decay is used.
            If `True`, `lr_ref` is the first available among
            `group["max_lr"]`, `group["initial_lr"]`, and the
            learning-rate passed at construction time.
            If a float is provided, it is used directly as `lr_ref`.
        
        centralize (bool, optional) [True]:
            --- Gradient Centralization ---
            
            Removes shared offsets from gradient slices before the range
            computation. This can improve generalization and training
            stability. Biases and other 1D parameters are not
            centralized.
        
        stabilize (bool, optional) [True]:
            --- Coefficient-of-Variation Trust Gating ---
            
            Computes a trust factor from the coefficient of variation of
            the per-slice range tensor, and then interpolates between the
            local range and a smoother global mean denominator.
        
        bf16_sr (bool or torch.Generator, optional) [True]:
            --- Stochastic Rounding for BFloat16 ---
            
            Improves BF16 training by using stochastic rounding instead
            of plain truncation when writing parameters back. This has
            no effect on non-BF16 parameters.
            
            BF16 parameters are promoted to `compute_dtype` (or to FP32
            if `compute_dtype` is `None`) for the update and then
            cast to FP32 for write-back. The result is stochastically
            rounded by adding uniform noise to the lower 16 bits of the
            FP32 representation before truncation to BF16.
            
            If `False`, BF16 stochastic rounding is disabled.
            If `True`, uses the default random-number generator.
            If a `torch.Generator` is provided instead of a boolean,
            treats `bf16_sr` as enabled and forwards that generator
            to `random_`. This is useful when you want reproducible
            stochastic rounding noise.
        
        compute_dtype (torch.dtype, str, or None, optional) [fp64]:
            --- Internal Compute Precision ---
            
            Promotes parameters and gradients to this dtype for the
            update step, then casts them back on write-back. Setting this
            to `None` disables promotion and computes in each parameter's
            native dtype, except that BF16 parameters still use FP32 when
            `bf16_sr` is enabled.
            
            FP64 is recommended because the intermediate range and
            division arithmetic benefits from the extra precision.
            
            In addition to passing a `torch.dtype` or `None`, the
            following strings are also valid: `float16`, `fp16`,
            `bfloat16`, `bf16`, `float32`, `fp32`, `float64`, `fp64`,
            `none`, and `null`.
    
    References:
        - Kingma, D. P. & Ba, J. (2014), Adam: A Method for
          Stochastic Optimization. arXiv:1412.6980
        - Loshchilov, I. & Hutter, F. (2017), Decoupled Weight Decay
          Regularization. arXiv:1711.05101
        - Hazan, E., Levy, K. Y., & Shalev-Shwartz, S. (2015),
          Beyond Convexity: Stochastic Quasi-Convex Optimization.
          arXiv:1507.02030
        - You, Y., Gitman, I., & Ginsburg, B. (2017), Large Batch
          Training of Convolutional Networks. arXiv:1708.03888
        - Yong, H., Huang, J., Hua, X. & Zhang, L. (2020), Gradient
          Centralization: A New Optimization Technique for Deep Neural
          Networks. arXiv:2004.01461
        - Zamirai, P., Zhang, J., Aberger, C. R. & De Sa, C. (2020),
          Revisiting BFloat16 Training. arXiv:2010.06192
    """
    def __init__(
        self,
        params,
        lr: float,
        *,
        weight_decay: float | None = 1e-4,
        wd_schedule: bool | float = False,
        centralize: bool = True,
        stabilize: bool = True,
        bf16_sr: bool | torch.Generator = True,
        compute_dtype: torch.dtype | str | None = "fp64"
    ):
        if lr < 0.0:
            raise ValueError(f"\nInvalid learning rate: {lr}") from None
        if weight_decay is not None and weight_decay < 0.0:
            raise ValueError(f"\nInvalid weight_decay: {weight_decay}") from None
        
        if isinstance(bf16_sr, torch.Generator):
            self.bf16_sr_gen = bf16_sr
            bf16_sr = True
        else:
            self.bf16_sr_gen = None
        
        if isinstance(compute_dtype, str):
            dtype_lookup: dict[str, torch.dtype | None] = {
                "float16": torch.float16, "fp16": torch.float16,
                "float32": torch.float32, "fp32": torch.float32,
                "float64": torch.float64, "fp64": torch.float64,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "none": None, "null": None
            }
            try:
                compute_dtype = dtype_lookup[compute_dtype.strip().lower()]
            except KeyError:
                raise ValueError(
                    f"\nInvalid compute_dtype string: {compute_dtype!r}.\n"
                    f"Valid options: {sorted(dtype_lookup)}"
                ) from None
        
        if bf16_sr and compute_dtype not in (torch.float32, torch.float64, None):
            raise ValueError(
                f"\nbf16_sr=True has no useful effect when compute_dtype is {compute_dtype}.\n"
                f"Use torch.float32, torch.float64, or None (same as fp32) instead."
            ) from None
        
        defaults = dict(
            lr=lr,
            centralize=centralize,
            stabilize=stabilize,
            weight_decay=weight_decay,
            wd_schedule=wd_schedule,
            bf16_sr=bf16_sr,
            compute_dtype=compute_dtype
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None) -> torch.Tensor | None:
        """Perform a single Rose optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            use_stabilize = group["stabilize"]
            use_centralize = group["centralize"]
            bf16_sr = group["bf16_sr"]
            compute_dtype = group["compute_dtype"]
            
            # --- Decoupled Weight Decay Factor ---
            weight_decay = group["weight_decay"]
            wd_schedule = group["wd_schedule"]
            group.setdefault("initial_lr", lr)
            
            # `wd_schedule` adapted from optimi's "Fully Decoupled Weight Decay":
            # https://optimi.benjaminwarner.dev/fully_decoupled_weight_decay/
            if weight_decay and wd_schedule:
                wd_lr = lr / (
                    wd_schedule if isinstance(wd_schedule, float)
                    else group.get("max_lr", group.get("initial_lr"))
                )
            else:
                wd_lr = lr
            
            wd_factor = None if not weight_decay else max(0.0, 1.0 - wd_lr * weight_decay)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Rose does not support sparse gradients")
                
                # --- Precision Handling ---
                use_bf16_sr = bf16_sr and p.dtype is torch.bfloat16
                fp32 = use_bf16_sr and not compute_dtype
                grad = p.grad.to(dtype=torch.float32 if fp32 else compute_dtype)
                param = p.to(dtype=torch.float32 if fp32 else compute_dtype)
                
                # --- Decoupled Multiplicative Weight Decay ---
                if wd_factor is not None:
                    param.mul_(wd_factor)
                
                # --- Active Axes: all axes except the first ---
                # Preserve the leading axis so that each slice receives its own scale.
                active_axes = tuple(range(1, grad.ndim))
                slice_numel = 1 if grad.ndim == 0 else grad[0].numel()
                
                if grad.numel() == 1:
                    # --- Scalar ---
                    # Floor chosen so that grad/(|grad|+floor) remains numerically
                    # meaningful near zero while still reducing to softsign-like
                    # behavior for larger gradients.
                    floor = torch.finfo(grad.dtype).resolution ** (1 / 3)
                    param.addcdiv_(grad, grad.abs().add_(floor), value=-lr)
                
                elif slice_numel == 1:
                    # --- Vectors / Degenerate Slices ---
                    # +1 damps small ranges
                    g_min, g_max = grad.aminmax()
                    denom = g_max.sub_(g_min).add_(1.0)
                    param.addcdiv_(grad, denom, value=-lr)
                
                else:
                    # --- Gradient Centralization ---
                    if use_centralize:
                        if grad.data_ptr() != p.grad.data_ptr():
                            grad.sub_(grad.mean(dim=active_axes, keepdim=True))
                        else:
                            grad = grad.sub(grad.mean(dim=active_axes, keepdim=True))
                    
                    # --- Per-slice Range: R = max(g) - min(g) over `active_axes` ---
                    # Reducing all non-leading dimensions at once yields one range
                    # value per leading-axis slice.
                    raw_scale = (
                        grad.amax(dim=active_axes, keepdim=True)
                        .sub_(grad.amin(dim=active_axes, keepdim=True))
                    )
                    
                    if use_stabilize:
                        # --- Coefficient-of-Variation Trust Gating ---
                        # Measures the self-consistency of per-slice ranges:
                        # Stable ranges preserve local detail.
                        # Noisy ranges use global mean for noise resistance.
                        std, mean = torch.std_mean(raw_scale, correction=0)
                        
                        # Trust factor:
                        # Higher when ranges are self-consistent.
                        # Lower when ranges are heterogeneous.
                        trust = mean.div(std.add_(mean).masked_fill_(mean == 0.0, 1.0))
                        
                        # Blend each local range with the smoother mean estimate.
                        denom = mean.lerp(raw_scale, trust)
                    else:
                        denom = raw_scale
                    
                    # --- Update: θ -= lr · g / D(g) ---
                    denom.masked_fill_(denom == 0.0, 1.0)  # SGD fallback
                    param.addcdiv_(grad, denom, value=-lr)
                
                if use_bf16_sr:
                    # --- BF16 stochastic rounding ---
                    # Inspired by Nerogar's code snippet: https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
                    
                    # P(round up) ∝ fractional distance → unbiased expectation
                    param = param.to(dtype=torch.float32)
                    p.copy_(
                        torch.empty_like(p, dtype=torch.int32)
                        .random_(0, 0x10000, generator=self.bf16_sr_gen)
                        .add_(param.view(dtype=torch.int32))
                        .bitwise_and_(-0x10000)
                        .view(dtype=torch.float32)
                    )
                
                elif param is not p:
                    p.copy_(param)
        
        return loss
