"""Public interface for the PreProcessing package."""

from __future__ import annotations

from importlib import import_module

from .Control import BFGS, Broyden, Control, TimeStepper, solver
from .Material import Material as _MaterialClass
from .Material import material
from .Model import Model
from .Step import Step as _StepClass

# Modules exposed as namespaces of helper classes/functions.
Boundary = import_module(".Boundary", __name__)
Constraints = import_module(".Constraints", __name__)
Contact = import_module(".Contact", __name__)
Globals = import_module(".Globals", __name__)
LoadData = import_module(".LoadData", __name__)
Loads = import_module(".Loads", __name__)
Output = import_module(".Output", __name__)
Rigid = import_module(".Rigid", __name__)


Material = _MaterialClass
Step = _StepClass

__all__ = [
    "Boundary",
    "Constraints",
    "Contact",
    "Globals",
    "LoadData",
    "Loads",
    "Material",
    "material",
    "Model",
    "Output",
    "Rigid",
    "Step",
    "Control",
    "solver",
    "TimeStepper",
    "BFGS",
    "Broyden",
]
