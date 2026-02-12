"""
Muon optimizer - Momentum Orthogonalized by Newton-schulz.

Implements momentum orthogonalization for improved training stability
and allows for higher learning rates compared to AdamW.

Reference: Based on momentum orthogonalization techniques
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional


class Muon8bit(Optimizer):
    """
    8-bit Muon optimizer with quantized momentum buffers.

    Stores momentum in 8-bit format to reduce memory by ~4x.
    Works with FSDP's flattened 1D parameters.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        newton_iters: int = 5,
        use_8bit: bool = True,
        block_size: int = 2048,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            newton_iters=newton_iters,
            use_8bit=use_8bit,
            block_size=block_size,
        )
        super().__init__(params, defaults)

        self.use_8bit = use_8bit
        print(f"  ✓ Using custom Muon optimizer ({'8-bit' if use_8bit else '32-bit'} momentum)")

    def _quantize_to_8bit(self, tensor, block_size=2048):
        """Quantize tensor to 8-bit with per-block scaling."""
        original_shape = tensor.shape
        flat = tensor.flatten()

        # Pad to multiple of block_size
        numel = flat.numel()
        if numel % block_size != 0:
            pad_size = block_size - (numel % block_size)
            flat = torch.nn.functional.pad(flat, (0, pad_size))

        # Reshape to blocks
        blocks = flat.view(-1, block_size)

        # Compute per-block scales (absmax)
        scales = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)

        # Quantize to int8 range [-127, 127]
        quantized = (blocks / scales * 127).round().to(torch.int8)

        return quantized, scales, original_shape, numel

    def _dequantize_from_8bit(self, quantized, scales, original_shape, original_numel):
        """Dequantize 8-bit tensor back to float."""
        # Dequantize
        dequantized = (quantized.float() / 127) * scales

        # Flatten and remove padding
        flat = dequantized.flatten()[:original_numel]

        return flat.view(original_shape)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            newton_iters = group['newton_iters']
            use_8bit = group['use_8bit']
            block_size = group['block_size']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                state = self.state[p]

                # Initialize or retrieve momentum buffer
                if use_8bit:
                    if 'momentum_quantized' not in state:
                        # First step: initialize with gradient
                        q, s, shape, numel = self._quantize_to_8bit(grad, block_size)
                        state['momentum_quantized'] = q
                        state['momentum_scales'] = s
                        state['momentum_shape'] = shape
                        state['momentum_numel'] = numel
                        buf = grad.clone()
                    else:
                        # Dequantize momentum buffer
                        buf = self._dequantize_from_8bit(
                            state['momentum_quantized'],
                            state['momentum_scales'],
                            state['momentum_shape'],
                            state['momentum_numel'],
                        ).to(grad.device)
                        # Update momentum
                        buf.mul_(momentum).add_(grad)
                else:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(grad)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)

                # Apply orthogonalization for 2D+ or large 1D (FSDP flattened)
                # For FSDP, we reshape 1D to approximate 2D
                if p.dim() >= 2 or (p.dim() == 1 and p.numel() > 1024):
                    original_shape = buf.shape

                    if p.dim() == 1:
                        # FSDP flattened: reshape to approximate square
                        numel = buf.numel()
                        # Find factors close to sqrt
                        sqrt_n = int(numel ** 0.5)
                        for i in range(sqrt_n, 0, -1):
                            if numel % i == 0:
                                rows, cols = i, numel // i
                                break
                        else:
                            rows, cols = 1, numel
                        buf_2d = buf.view(rows, cols)
                    elif p.dim() == 2:
                        buf_2d = buf
                    else:
                        buf_2d = buf.reshape(buf.shape[0], -1)

                    # Newton-Schulz orthogonalization
                    buf_ortho = self._newton_schulz(buf_2d, newton_iters)

                    if p.dim() == 1:
                        buf = buf_ortho.view(original_shape)
                    elif p.dim() == 2:
                        buf = buf_ortho
                    else:
                        buf = buf_ortho.reshape(original_shape)

                # Re-quantize updated momentum
                if use_8bit:
                    q, s, shape, numel = self._quantize_to_8bit(buf, block_size)
                    state['momentum_quantized'] = q
                    state['momentum_scales'] = s
                    state['momentum_shape'] = shape
                    state['momentum_numel'] = numel

                # Apply update
                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf

                p.add_(update, alpha=-lr)

        return loss

    @staticmethod
    def _newton_schulz(matrix, num_iters=5):
        """Newton-Schulz orthogonalization."""
        m, n = matrix.shape
        Y = matrix / (matrix.norm() + 1e-7)

        if m <= n:
            for _ in range(num_iters):
                YYT = Y @ Y.T
                Y = 1.5 * Y - 0.5 * (YYT @ Y)
        else:
            for _ in range(num_iters):
                YTY = Y.T @ Y
                Y = 1.5 * Y - 0.5 * (Y @ YTY)

        return Y

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()


class Muon(Optimizer):
    """
    Muon optimizer with momentum orthogonalization.

    Key features:
    - Orthogonalized momentum updates for better gradient flow
    - Higher learning rates possible (typically 3-10x AdamW)
    - Better handling of ill-conditioned problems
    - Built-in weight decay

    Args:
        params: Model parameters
        lr: Learning rate (typically 1e-3 to 3e-3, higher than AdamW)
        momentum: Momentum coefficient (default: 0.95)
        weight_decay: Weight decay coefficient (default: 0.01)
        nesterov: Whether to use Nesterov momentum (default: True)
        backend: Backend for orthogonalization ('newton' or 'qr', default: 'newton')
        newton_iters: Number of Newton-Schulz iterations (default: 5)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        backend: str = 'newton',
        newton_iters: int = 5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            backend=backend,
            newton_iters=newton_iters,
        )
        super().__init__(params, defaults)

        print(f"Initialized Muon optimizer:")
        print(f"  Learning rate: {lr:.2e}")
        print(f"  Momentum: {momentum}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Nesterov: {nesterov}")
        print(f"  Backend: {backend}")
        if backend == 'newton':
            print(f"  Newton iterations: {newton_iters}")

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: Optional closure to re-evaluate the model

        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            backend = group['backend']
            newton_iters = group['newton_iters']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay (decoupled, like AdamW)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Get or initialize momentum buffer
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']

                # Update momentum: buf = momentum * buf + grad
                buf.mul_(momentum).add_(grad, alpha=1.0)

                # Orthogonalize momentum for matrix-like parameters
                # For 2D+ tensors, apply orthogonalization
                if p.dim() >= 2:
                    # Reshape to 2D matrix for orthogonalization
                    original_shape = buf.shape

                    # For Conv weights (4D: out, in, h, w), reshape to (out, in*h*w)
                    # For Linear weights (2D: out, in), keep as is
                    if p.dim() == 2:
                        buf_2d = buf
                    else:
                        # Flatten all dims except first
                        buf_2d = buf.reshape(buf.shape[0], -1)

                    # Apply orthogonalization
                    if backend == 'qr':
                        # QR decomposition (more expensive but exact)
                        if buf_2d.shape[0] <= buf_2d.shape[1]:
                            # More columns than rows: orthogonalize rows
                            Q, R = torch.linalg.qr(buf_2d.T)
                            buf_ortho = Q.T
                        else:
                            # More rows than columns: orthogonalize columns
                            Q, R = torch.linalg.qr(buf_2d)
                            buf_ortho = Q
                    else:
                        # Newton-Schulz iteration (faster approximation)
                        buf_ortho = self._newton_schulz_orthogonalize(
                            buf_2d, newton_iters
                        )

                    # Reshape back
                    if p.dim() == 2:
                        buf.copy_(buf_ortho)
                    else:
                        buf.copy_(buf_ortho.reshape(original_shape))

                # Apply update
                if nesterov:
                    # Nesterov momentum: use (grad + momentum * buf)
                    # where buf is the updated momentum buffer
                    update = grad.add(buf, alpha=momentum)
                else:
                    # Standard momentum: use buf
                    update = buf

                # Update parameters
                p.add_(update, alpha=-lr)

        return loss

    @staticmethod
    def _newton_schulz_orthogonalize(
        matrix: torch.Tensor,
        num_iters: int = 5,
    ) -> torch.Tensor:
        """
        Orthogonalize matrix using Newton-Schulz iteration.

        This is a fast iterative method to approximate orthogonalization.
        Converges to Q where Q @ Q.T ≈ I (for tall matrices)
        or Q.T @ Q ≈ I (for wide matrices).

        Args:
            matrix: 2D tensor to orthogonalize
            num_iters: Number of iterations (5-10 usually sufficient)

        Returns:
            Approximately orthogonalized matrix
        """
        m, n = matrix.shape

        if m <= n:
            # Wide matrix: orthogonalize rows
            # Want Y @ Y.T ≈ I (rows are orthonormal)
            # Initialize with normalized matrix
            Y = matrix / (matrix.norm() + 1e-7)

            for _ in range(num_iters):
                # Newton-Schulz iteration for row orthogonalization
                # Y = 1.5*Y - 0.5*(Y@Y.T)@Y
                YYT = Y @ Y.T  # (m, m)
                Y = 1.5 * Y - 0.5 * (YYT @ Y)  # (m, n)

            return Y
        else:
            # Tall matrix: orthogonalize columns
            # Want Y.T @ Y ≈ I (columns are orthonormal)
            Y = matrix / (matrix.norm() + 1e-7)

            for _ in range(num_iters):
                # Newton-Schulz iteration for column orthogonalization
                # Y = 1.5*Y - 0.5*Y@(Y.T@Y)
                YTY = Y.T @ Y  # (n, n)
                Y = 1.5 * Y - 0.5 * (Y @ YTY)  # (m, n)

            return Y

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients (optionally set to None for memory efficiency)."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()


def get_muon_optimizer(
    model,
    lr: float = 1e-3,
    momentum: float = 0.95,
    weight_decay: float = 0.01,
):
    """
    Create hybrid Muon + AdamW optimizer using PyTorch's built-in Muon.

    Muon paper approach:
    - Use Muon for 2D parameters (weight matrices in hidden layers)
    - Use AdamW for 1D parameters (biases, norms, embeddings) and first/last layers

    Args:
        model: Model to optimize
        lr: Learning rate for Muon
        momentum: Momentum coefficient for Muon
        weight_decay: Weight decay coefficient

    Returns:
        Hybrid optimizer using Muon + AdamW
    """
    import torch
    import os

    # Separate parameters for Muon (2D) vs AdamW (1D, embeddings, first/last layers)
    muon_params = []
    adamw_params = []

    # Check if running under FSDP (params may be flattened to 1D)
    is_fsdp = bool(os.environ.get("WORLD_SIZE") or os.environ.get("LOCAL_RANK"))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine if this should use Muon based on name patterns
        # This works even when FSDP flattens params to 1D
        name_lower = name.lower()

        # Exclude from Muon:
        # - Embeddings, LM heads, biases, norms
        # - Non-weight parameters
        is_excluded = (
            'embedding' in name_lower or
            'embed' in name_lower or
            'lm_head' in name_lower or
            'head' in name_lower or
            'bias' in name_lower or
            'norm' in name_lower or
            'layernorm' in name_lower or
            'ln_' in name_lower or
            'scale' in name_lower
        )

        # Include in Muon: weight matrices (detected by .weight in name)
        is_weight_matrix = '.weight' in name_lower and not is_excluded

        if is_fsdp:
            # Under FSDP: use name-based detection since dims are flattened
            if is_weight_matrix:
                muon_params.append(param)
            else:
                adamw_params.append(param)
        else:
            # Not FSDP: use original dimension check + name patterns
            if param.dim() == 2 and is_weight_matrix:
                muon_params.append(param)
            else:
                adamw_params.append(param)

    print(f"\nHybrid Muon + AdamW optimizer:")
    print(f"  FSDP mode: {is_fsdp} (using {'name-based' if is_fsdp else 'dimension-based'} detection)")
    print(f"  Muon params (2D weight matrices): {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW params (embed/head/bias/norm): {sum(p.numel() for p in adamw_params):,}")

    # Use custom Muon (works with FSDP's flattened 1D params, unlike torch.optim.Muon)
    if len(muon_params) > 0:
        muon_opt = Muon8bit(
            muon_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        muon_opt = None

    if len(adamw_params) > 0:
        # Try 8-bit AdamW if bitsandbytes available
        try:
            import bitsandbytes as bnb
            adamw_opt = bnb.optim.AdamW8bit(
                adamw_params,
                lr=lr * 0.3,
                betas=(0.9, 0.95),
                weight_decay=weight_decay,
            )
            print(f"  ✓ Using 8-bit AdamW (bitsandbytes)")
        except ImportError:
            adamw_opt = torch.optim.AdamW(
                adamw_params,
                lr=lr * 0.3,
                betas=(0.9, 0.95),
                weight_decay=weight_decay,
            )
            print(f"  ✓ Using standard AdamW (bitsandbytes not available)")
    else:
        adamw_opt = None

    # Return wrapper that calls both
    class HybridOptimizer:
        def __init__(self, muon_opt, adamw_opt):
            self.muon_opt = muon_opt
            self.adamw_opt = adamw_opt

        def step(self):
            if self.muon_opt:
                self.muon_opt.step()
            if self.adamw_opt:
                self.adamw_opt.step()

        def zero_grad(self, set_to_none=True):
            if self.muon_opt:
                self.muon_opt.zero_grad(set_to_none)
            if self.adamw_opt:
                self.adamw_opt.zero_grad(set_to_none)

        def state_dict(self):
            return {
                'muon': self.muon_opt.state_dict() if self.muon_opt else None,
                'adamw': self.adamw_opt.state_dict() if self.adamw_opt else None,
            }

        def load_state_dict(self, state_dict):
            if self.muon_opt and state_dict.get('muon'):
                self.muon_opt.load_state_dict(state_dict['muon'])
            if self.adamw_opt and state_dict.get('adamw'):
                self.adamw_opt.load_state_dict(state_dict['adamw'])

        @property
        def param_groups(self):
            groups = []
            if self.muon_opt:
                groups.extend(self.muon_opt.param_groups)
            if self.adamw_opt:
                groups.extend(self.adamw_opt.param_groups)
            return groups

    return HybridOptimizer(muon_opt, adamw_opt)


if __name__ == "__main__":
    # Test Muon optimizer
    print("Testing Muon optimizer...")

    # Create simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.LayerNorm(128),
        torch.nn.Linear(128, 10),
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = get_muon_optimizer(
        model,
        lr=1e-3,
        momentum=0.95,
        weight_decay=0.01,
    )

    # Test optimization step
    print("\nTesting optimization step...")
    x = torch.randn(32, 128)
    target = torch.randint(0, 10, (32,))

    # Forward
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, target)
    print(f"  Initial loss: {loss.item():.4f}")

    # Backward
    loss.backward()

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Check parameters updated
    output2 = model(x)
    loss2 = torch.nn.functional.cross_entropy(output2, target)
    print(f"  Loss after step: {loss2.item():.4f}")

    assert loss.item() != loss2.item(), "Parameters should have changed"
    print("  ✓ Parameters updated")

    # Test multiple steps
    print("\nTesting 100 optimization steps...")
    for i in range(100):
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 20 == 0:
            print(f"  Step {i+1}: loss = {loss.item():.4f}")

    print("  ✓ Multiple steps work")

    # Test orthogonalization
    print("\nTesting orthogonalization...")
    matrix = torch.randn(64, 128)
    ortho = Muon._newton_schulz_orthogonalize(matrix, num_iters=10)

    # Check orthogonality: Q @ Q.T should be close to identity
    product = ortho @ ortho.T
    identity = torch.eye(64)
    error = (product - identity).abs().max().item()
    print(f"  Max orthogonality error: {error:.6f}")
    assert error < 0.1, f"Orthogonalization error too large: {error}"
    print("  ✓ Orthogonalization works")

    print("\n✓ All Muon optimizer tests passed!")
