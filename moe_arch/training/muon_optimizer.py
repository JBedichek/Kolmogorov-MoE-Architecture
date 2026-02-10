"""
Muon optimizer - Momentum Orthogonalized by Newton-schulz.

Implements momentum orthogonalization for improved training stability
and allows for higher learning rates compared to AdamW.

Reference: Based on momentum orthogonalization techniques
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional


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

    # Separate parameters for Muon (2D) vs AdamW (1D, embeddings, first/last layers)
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Use AdamW for:
        # - Non-2D parameters (1D biases/norms, 3D conv1d, 4D conv2d)
        # - Embeddings (first layer)
        # - LM heads (last layer)
        # PyTorch's built-in Muon ONLY supports exactly 2D parameters
        if (param.dim() != 2 or
            'embedding' in name.lower() or
            'lm_head' in name.lower() or
            'bias' in name.lower() or
            'norm' in name.lower()):
            adamw_params.append(param)
        else:
            # Use Muon for exactly 2D weight matrices only
            muon_params.append(param)

    print(f"\nHybrid Muon + AdamW optimizer (using PyTorch built-in Muon):")
    print(f"  Muon params (exactly 2D weights): {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW params (1D/3D+/embed/heads): {sum(p.numel() for p in adamw_params):,}")

    # Create hybrid optimizer using PyTorch's built-in Muon
    if len(muon_params) > 0:
        muon_opt = torch.optim.Muon(
            muon_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        print(f"  ✓ Using PyTorch built-in Muon optimizer")
    else:
        muon_opt = None

    if len(adamw_params) > 0:
        adamw_opt = torch.optim.AdamW(
            adamw_params,
            lr=lr * 0.3,  # Lower LR for AdamW (typically 3x lower)
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )
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
