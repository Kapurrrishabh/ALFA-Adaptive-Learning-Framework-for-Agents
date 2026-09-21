"""AdamW with decoupled weight decay (Loshchilov & Hutter, 2019)."""

from ..backend import xp


class AdamW:
    def __init__(self, parameters, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.parameters = [p for p in parameters if p.requires_grad]
        if not self.parameters:
            raise ValueError("AdamW was given no trainable parameters")
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.steps = 0
        self._moment = [xp.zeros_like(p.data) for p in self.parameters]
        self._velocity = [xp.zeros_like(p.data) for p in self.parameters]

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        self.steps += 1
        bias_correction1 = 1.0 - self.beta1**self.steps
        bias_correction2 = 1.0 - self.beta2**self.steps

        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            self._moment[index] = (
                self.beta1 * self._moment[index] + (1.0 - self.beta1) * gradient
            )
            self._velocity[index] = self.beta2 * self._velocity[index] + (
                1.0 - self.beta2
            ) * (gradient * gradient)

            step_size = self.lr / bias_correction1
            denominator = xp.sqrt(self._velocity[index] / bias_correction2) + self.eps
            update = step_size * self._moment[index] / denominator

            # Biases and LayerNorm gains are the only 1-D parameters, and decaying them hurts.
            if self.weight_decay and parameter.data.ndim > 1:
                update = update + self.lr * self.weight_decay * parameter.data

            parameter.data = parameter.data - update
