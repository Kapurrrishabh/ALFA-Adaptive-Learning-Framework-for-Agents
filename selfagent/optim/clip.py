"""Global gradient-norm clipping."""

from ..backend import xp


def clip_grad_norm(parameters, max_norm):
    """Scales all gradients down together. Returns the norm before clipping, for logging."""
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return 0.0
    total_norm = float(xp.sqrt(sum(float((g * g).sum()) for g in gradients)))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad = parameter.grad * scale
    return total_norm
