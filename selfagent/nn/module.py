"""Base class for every layer. Owns parameter discovery, train/eval mode and checkpointing."""

from .. import backend
from ..backend import xp
from ..autograd.tensor import Tensor


class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_children", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                f"{type(self).__name__} assigned '{name}' before calling super().__init__()"
            )
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._children[name] = value
        object.__setattr__(self, name, value)

    def named_parameters(self, prefix=""):
        for name, parameter in self._parameters.items():
            yield prefix + name, parameter
        for name, child in self._children.items():
            yield from child.named_parameters(prefix + name + ".")

    def parameters(self):
        return [parameter for _, parameter in self.named_parameters()]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = None

    def train(self, mode=True):
        self.training = mode
        for child in self._children.values():
            child.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        return {name: parameter.data for name, parameter in self.named_parameters()}

    def load_state_dict(self, state):
        own = dict(self.named_parameters())
        missing = sorted(set(own) - set(state))
        unexpected = sorted(set(state) - set(own))
        if missing or unexpected:
            raise KeyError(
                f"checkpoint does not match the model. missing={missing} unexpected={unexpected}"
            )
        for name, parameter in own.items():
            incoming = xp.asarray(state[name], dtype=backend.dtype())
            if incoming.shape != parameter.data.shape:
                raise ValueError(
                    f"'{name}' expects shape {parameter.data.shape}, checkpoint has {incoming.shape}"
                )
            parameter.data = incoming

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} has no forward()")


class ModuleList(Module):
    """A sequence of children, since plain lists are invisible to __setattr__."""

    def __init__(self, modules):
        super().__init__()
        self._items = list(modules)
        for index, module in enumerate(self._items):
            self._children[str(index)] = module

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]
