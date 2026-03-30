from __future__ import annotations

import os
import warnings


_ROCM_DROPOUT_WARNING_EMITTED = False


def _resolve_dropout(dropout: float, num_layers: int) -> tuple[float, float]:
    """Return (head_dropout, recurrent_dropout) with ROCm-safe defaults."""
    import torch
    global _ROCM_DROPOUT_WARNING_EMITTED

    head_dropout = float(dropout)
    recurrent_dropout = 0.1 if num_layers > 1 else 0.0
    disable_rocm_dropout = os.environ.get("FORECAST_ROCM_DISABLE_DROPOUT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if getattr(torch.version, "hip", None) and disable_rocm_dropout and (head_dropout > 0.0 or recurrent_dropout > 0.0):
        if not _ROCM_DROPOUT_WARNING_EMITTED:
            warnings.warn(
                "ROCm detected; disabling model dropout to avoid MIOpen HIPRTC rocrand build failures. "
                "Set FORECAST_ROCM_DISABLE_DROPOUT=0 to re-enable dropout.",
                RuntimeWarning,
            )
            _ROCM_DROPOUT_WARNING_EMITTED = True
        return 0.0, 0.0

    return head_dropout, recurrent_dropout


class _GRUNet:
    """Thin wrapper around a PyTorch GRU module."""

    def __init__(
        self,
        hidden: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        import torch
        import torch.nn as nn

        _ACT_MAP = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
        }
        _act = _ACT_MAP.get(activation, nn.ReLU())
        _drop, _recurrent_drop = _resolve_dropout(dropout, num_layers)

        class _Net(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.gru = nn.GRU(
                    input_size=1,
                    hidden_size=hidden,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=_recurrent_drop,
                )
                self_.act = _act
                self_.drop = nn.Dropout(_drop) if _drop > 0.0 else nn.Identity()
                self_.fc = nn.Linear(hidden, 1)

            def forward(self_, x):
                out, _ = self_.gru(x)
                h = self_.act(out[:, -1])
                h = self_.drop(h)
                return self_.fc(h)

        self.device = device
        self.net = _Net().to(device)
        self.torch = torch
        self.hidden = hidden

    def parameters(self):
        return self.net.parameters()

    def train_mode(self):
        self.net.train()

    def eval_mode(self):
        self.net.eval()

    def __call__(self, x):
        return self.net(x)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


class _LSTMNet:
    """Thin wrapper around a PyTorch LSTM module (same interface as _GRUNet)."""

    def __init__(
        self,
        hidden: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        import torch
        import torch.nn as nn

        _ACT_MAP = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
        }
        _act = _ACT_MAP.get(activation, nn.ReLU())
        _drop, _recurrent_drop = _resolve_dropout(dropout, num_layers)

        class _Net(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.lstm = nn.LSTM(
                    input_size=1,
                    hidden_size=hidden,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=_recurrent_drop,
                )
                self_.act = _act
                self_.drop = nn.Dropout(_drop) if _drop > 0.0 else nn.Identity()
                self_.fc = nn.Linear(hidden, 1)

            def forward(self_, x):
                out, _ = self_.lstm(x)
                h = self_.act(out[:, -1])
                h = self_.drop(h)
                return self_.fc(h)

        self.device = device
        self.net = _Net().to(device)
        self.torch = torch
        self.hidden = hidden

    def parameters(self):
        return self.net.parameters()

    def train_mode(self):
        self.net.train()

    def eval_mode(self):
        self.net.eval()

    def __call__(self, x):
        return self.net(x)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


class _RNNNet:
    """Thin wrapper around a PyTorch Elman RNN module (same interface as _GRUNet)."""

    def __init__(
        self,
        hidden: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        import torch
        import torch.nn as nn

        _ACT_MAP = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
        }
        _act = _ACT_MAP.get(activation, nn.ReLU())
        _drop, _recurrent_drop = _resolve_dropout(dropout, num_layers)

        class _Net(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.rnn = nn.RNN(
                    input_size=1,
                    hidden_size=hidden,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=_recurrent_drop,
                    nonlinearity="tanh",
                )
                self_.act = _act
                self_.drop = nn.Dropout(_drop) if _drop > 0.0 else nn.Identity()
                self_.fc = nn.Linear(hidden, 1)

            def forward(self_, x):
                out, _ = self_.rnn(x)
                h = self_.act(out[:, -1])
                h = self_.drop(h)
                return self_.fc(h)

        self.device = device
        self.net = _Net().to(device)
        self.torch = torch
        self.hidden = hidden

    def parameters(self):
        return self.net.parameters()

    def train_mode(self):
        self.net.train()

    def eval_mode(self):
        self.net.eval()

    def __call__(self, x):
        return self.net(x)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


class _MLPNet:
    """Simple MLP on flattened sequence windows."""

    def __init__(
        self,
        hidden: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        import torch
        import torch.nn as nn

        _ACT_MAP = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
        }
        _act = _ACT_MAP.get(activation, nn.ReLU())
        _drop, _ = _resolve_dropout(dropout, num_layers)

        class _Net(nn.Module):
            def __init__(self_):
                super().__init__()
                layers = [nn.Flatten(), nn.LazyLinear(hidden)]
                depth = max(1, int(num_layers))
                for _ in range(depth):
                    layers.append(_act)
                    if _drop > 0.0:
                        layers.append(nn.Dropout(_drop))
                    layers.append(nn.Linear(hidden, hidden))
                layers.append(_act)
                if _drop > 0.0:
                    layers.append(nn.Dropout(_drop))
                layers.append(nn.Linear(hidden, 1))
                self_.net = nn.Sequential(*layers)

            def forward(self_, x):
                return self_.net(x)

        self.device = device
        self.net = _Net().to(device)
        self.torch = torch
        self.hidden = hidden

    def parameters(self):
        return self.net.parameters()

    def train_mode(self):
        self.net.train()

    def eval_mode(self):
        self.net.eval()

    def __call__(self, x):
        return self.net(x)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


class _TCNNet:
    """Dilated causal-convolution style temporal network."""

    def __init__(
        self,
        hidden: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        device: str = "cpu",
    ):
        import torch
        import torch.nn as nn

        _ACT_MAP = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
        }
        _act = _ACT_MAP.get(activation, nn.ReLU())
        _drop, _ = _resolve_dropout(dropout, num_layers)

        class _TemporalBlock(nn.Module):
            def __init__(self_, in_ch: int, out_ch: int, dilation: int):
                super().__init__()
                padding = 2 * dilation
                self_.conv = nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    dilation=dilation,
                    padding=padding,
                )
                self_.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
                self_.drop = nn.Dropout(_drop) if _drop > 0.0 else nn.Identity()
                self_.act = _act

            def forward(self_, x):
                y = self_.conv(x)
                y = y[..., : x.shape[-1]]
                y = self_.act(y)
                y = self_.drop(y)
                r = self_.proj(x)
                return self_.act(y + r)

        class _Net(nn.Module):
            def __init__(self_):
                super().__init__()
                blocks = []
                in_ch = 1
                depth = max(1, int(num_layers))
                for i in range(depth):
                    d = 2**i
                    blocks.append(_TemporalBlock(in_ch, hidden, d))
                    in_ch = hidden
                self_.blocks = nn.ModuleList(blocks)
                self_.head = nn.Linear(hidden, 1)

            def forward(self_, x):
                y = x.transpose(1, 2)
                for block in self_.blocks:
                    y = block(y)
                h = y[:, :, -1]
                return self_.head(h)

        self.device = device
        self.net = _Net().to(device)
        self.torch = torch
        self.hidden = hidden

    def parameters(self):
        return self.net.parameters()

    def train_mode(self):
        self.net.train()

    def eval_mode(self):
        self.net.eval()

    def __call__(self, x):
        return self.net(x)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


_MODEL_CLASSES = {
    "gru": _GRUNet,
    "lstm": _LSTMNet,
    "rnn": _RNNNet,
    "mlp": _MLPNet,
    "tcn": _TCNNet,
}


def average_model_weights(state_dicts: list, aggregation_method: str = "mean") -> dict:
    """Federated aggregation: combine model weights via mean or median.

    Parameters
    ----------
    state_dicts : list
        List of model state_dict objects (each a dict of parameter tensors).
    aggregation_method : str
        Either "mean" or "median" for aggregation.

    Returns
    -------
    dict
        Aggregated state_dict where each parameter is the mean/median of inputs.

    Raises
    ------
    ValueError
        If state_dicts is empty or all items are None.
    """
    import torch

    if not state_dicts or all(s is None for s in state_dicts):
        raise ValueError("state_dicts cannot be empty or all None.")

    state_dicts = [s for s in state_dicts if s is not None]
    if not state_dicts:
        raise ValueError("No valid state_dicts provided after filtering None.")

    aggregated = {}
    for key in state_dicts[0].keys():
        stacked = torch.stack([state[key].float() for state in state_dicts])

        if aggregation_method.lower() == "median":
            agg_tensor = torch.median(stacked, dim=0).values
        else:  # "mean" or default
            agg_tensor = stacked.mean(dim=0)

        aggregated[key] = agg_tensor

    return aggregated
