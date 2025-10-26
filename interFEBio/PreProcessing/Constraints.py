"""
This module deals with boundary conditions
https://github.com/febiosoftware/FEBioStudio/blob/develop/FEMLib/FEBoundaryCondition.cpp

"""

import xml.etree.cElementTree as ET
from typing import Literal

from .dataClass import _var, sectionAttribs


class PrestrainGradient:
    _opts = Literal[0, 1]

    def __init__(
        self,
        name: str,
        update: int | bool | str = True,
        tolerance: float | str = 0,
        min_iters: int | str = 0,
        max_iters: int | str = -1,
        attributes: dict | None = None,
    ):
        if attributes is None:
            attributes = {}
        self.name = name
        self.atr = sectionAttribs()
        self.atr.addAttrib(_var("update", update, attributes))
        self.atr.addAttrib(_var("tolerance", tolerance, attributes))
        self.atr.addAttrib(_var("min_iters", min_iters, attributes))
        self.atr.addAttrib(_var("max_iters", max_iters, attributes))

    def tree(self):
        tree = ET.Element("constraint", type="prestrain", name=self.name)
        self.atr.fillTree(tree)
        return tree
