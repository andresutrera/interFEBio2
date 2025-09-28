"""FEBio interface class exposure package."""

__version__ = "0.1.0"

from importlib import import_module

from .config import FEBIO_ROOT, resolve_febio_path

__all__ = [
    "__version__",
    "extractor",
    "generator",
    "febio_bindings",
    "FEBIO_ROOT",
    "resolve_febio_path",
]


_LAZY_MODULES = {"extractor", "generator", "febio_bindings"}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(name)
