class ConfigError(ValueError):
    """Raised when a RunSpec violates its declared contract."""


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot be executed faithfully."""
