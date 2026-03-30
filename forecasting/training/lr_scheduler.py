"""Learning rate scheduling strategies for model training."""

from abc import ABC, abstractmethod
from typing import Optional


class LRScheduler(ABC):
    """Base class for learning rate schedulers."""

    def __init__(self, initial_lr: float):
        """Initialize scheduler with initial learning rate.
        
        Args:
            initial_lr: The starting learning rate value.
        """
        self.initial_lr = initial_lr
        self.current_epoch = 0

    @abstractmethod
    def get_lr(self) -> float:
        """Get learning rate for the current epoch.
        
        Returns:
            Learning rate value for the current epoch.
        """
        pass

    def step(self):
        """Advance to the next epoch."""
        self.current_epoch += 1


class ConstantLR(LRScheduler):
    """Constant (static) learning rate."""

    def get_lr(self) -> float:
        """Return constant learning rate."""
        return self.initial_lr


class StepDecayLR(LRScheduler):
    """Step decay: reduce LR by a factor every N epochs.
    
    Args:
        initial_lr: Starting learning rate.
        step_size: Number of epochs before reducing LR.
        gamma: Multiplicative factor to apply at each step (default: 0.1).
    
    Example:
        LR = 0.01 initially
        After 10 epochs: LR = 0.01 * 0.1 = 0.001
        After 20 epochs: LR = 0.001 * 0.1 = 0.0001
    """

    def __init__(self, initial_lr: float, step_size: int = 10, gamma: float = 0.1):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> float:
        """Return decayed learning rate."""
        return self.initial_lr * (self.gamma ** (self.current_epoch // self.step_size))


class ExponentialDecayLR(LRScheduler):
    """Exponential decay: LR = initial_lr * exp(-decay_rate * epoch).
    
    Args:
        initial_lr: Starting learning rate.
        decay_rate: Rate of exponential decay per epoch (default: 0.01).
    """

    def __init__(self, initial_lr: float, decay_rate: float = 0.01):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate

    def get_lr(self) -> float:
        """Return exponentially decayed learning rate."""
        import math
        return self.initial_lr * math.exp(-self.decay_rate * self.current_epoch)


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing: smoothly decrease LR from initial to min_lr following a cosine curve.
    
    Args:
        initial_lr: Starting learning rate.
        total_epochs: Total number of epochs for training.
        min_lr: Minimum learning rate (default: 1e-6).
    """

    def __init__(
        self, initial_lr: float, total_epochs: int, min_lr: float = 1e-6
    ):
        super().__init__(initial_lr)
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def get_lr(self) -> float:
        """Return cosine-annealed learning rate."""
        import math
        # Cosine annealing formula
        return (
            self.min_lr
            + 0.5
            * (self.initial_lr - self.min_lr)
            * (1 + math.cos(math.pi * self.current_epoch / self.total_epochs))
        )


class LinearWarmupThenDecayLR(LRScheduler):
    """Linear warmup followed by decay.
    
    Args:
        initial_lr: Target learning rate after warmup.
        total_epochs: Total number of epochs for training.
        warmup_epochs: Number of epochs to warm up (default: 5).
        decay_type: Type of decay - 'step', 'exponential', or 'cosine' (default: 'cosine').
    """

    def __init__(
        self,
        initial_lr: float,
        total_epochs: int,
        warmup_epochs: int = 5,
        decay_type: str = "cosine",
    ):
        super().__init__(initial_lr)
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.decay_type = decay_type.lower()

    def get_lr(self) -> float:
        """Return warmup-then-decay learning rate."""
        import math

        if self.current_epoch < self.warmup_epochs:
            # Linear warmup: gradually increase from ~0 to initial_lr
            # Use (current_epoch + 1) to ensure non-zero LR at epoch 0
            return self.initial_lr * ((self.current_epoch + 1) / self.warmup_epochs)

        # Decay phase
        decay_epoch = self.current_epoch - self.warmup_epochs
        decay_total = self.total_epochs - self.warmup_epochs

        if self.decay_type == "step":
            # Step decay during decay phase
            return self.initial_lr * (0.1 ** (decay_epoch // max(1, decay_total // 3)))
        elif self.decay_type == "exponential":
            # Exponential decay
            return self.initial_lr * math.exp(-0.01 * decay_epoch)
        else:  # cosine
            # Cosine annealing
            return self.initial_lr * 0.5 * (
                1 + math.cos(math.pi * decay_epoch / decay_total)
            )


def get_scheduler(
    scheduler_type: str,
    initial_lr: float,
    epochs: int,
    **kwargs,
) -> LRScheduler:
    """Factory function to get the appropriate learning rate scheduler.
    
    Args:
        scheduler_type: Type of scheduler ('constant', 'step', 'exponential', 
                       'cosine', or 'warmup_decay').
        initial_lr: Starting learning rate.
        epochs: Total number of epochs.
        **kwargs: Additional arguments for specific schedulers.
    
    Returns:
        Instantiated LRScheduler object.
    
    Raises:
        ValueError: If scheduler_type is not recognized.
    """
    scheduler_type = scheduler_type.lower().strip()

    if scheduler_type == "constant":
        return ConstantLR(initial_lr)
    elif scheduler_type == "step":
        step_size = kwargs.get("step_size", max(1, epochs // 4))
        gamma = kwargs.get("gamma", 0.1)
        return StepDecayLR(initial_lr, step_size=step_size, gamma=gamma)
    elif scheduler_type == "exponential":
        decay_rate = kwargs.get("decay_rate", 0.01)
        return ExponentialDecayLR(initial_lr, decay_rate=decay_rate)
    elif scheduler_type == "cosine":
        min_lr = kwargs.get("min_lr", 1e-6)
        return CosineAnnealingLR(initial_lr, total_epochs=epochs, min_lr=min_lr)
    elif scheduler_type == "warmup_decay":
        warmup_epochs = kwargs.get("warmup_epochs", max(1, epochs // 10))
        decay_type = kwargs.get("decay_type", "cosine")
        return LinearWarmupThenDecayLR(
            initial_lr,
            total_epochs=epochs,
            warmup_epochs=warmup_epochs,
            decay_type=decay_type,
        )
    else:
        raise ValueError(
            f"Unknown scheduler type '{scheduler_type}'. "
            f"Choose from: 'constant', 'step', 'exponential', 'cosine', 'warmup_decay'."
        )
