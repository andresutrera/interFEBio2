"""
This module deals with boundary conditions
https://github.com/febiosoftware/FEBioStudio/blob/develop/FEMLib/FEBoundaryCondition.cpp

"""

import xml.etree.cElementTree as ET
from typing import Literal

from .dataClass import _var, sectionAttribs


class rigidFixed:
    _dof = Literal["x", "y", "z"]

    def __init__(
        self,
        name: str,
        material: str,
        Rx_dof: int | str | bool = 0,
        Ry_dof: int | str | bool = 0,
        Rz_dof: int | str | bool = 0,
        Ru_dof: int | str | bool = 0,
        Rv_dof: int | str | bool = 0,
        Rw_dof: int | str | bool = 0,
    ):
        self.name = name
        self.atr = sectionAttribs()
        self.atr.addAttrib(_var("rb", material))
        self.atr.addAttrib(_var("Rx_dof", Rx_dof))
        self.atr.addAttrib(_var("Ry_dof", Ry_dof))
        self.atr.addAttrib(_var("Rz_dof", Rz_dof))
        self.atr.addAttrib(_var("Ru_dof", Ru_dof))
        self.atr.addAttrib(_var("Rv_dof", Rv_dof))
        self.atr.addAttrib(_var("Rw_dof", Rw_dof))

    def tree(self):
        tree = ET.Element("rigid_bc", name=self.name, type="rigid_fixed")
        self.atr.fillTree(tree)
        return tree


class rigidDisplacement:
    _dof = Literal["x", "y", "z"]

    def __init__(
        self,
        name: str,
        material: str,
        dof: _dof,
        value: float = 1.0,
        relative: bool = False,
        attributes: dict | None = None,
    ):
        if attributes is None:
            attributes = {}
        self.name = name
        self.atr = sectionAttribs()
        self.atr.addAttrib(_var("rb", material))
        self.atr.addAttrib(_var("dof", dof))
        self.atr.addAttrib(_var("value", value, attributes))
        self.atr.addAttrib(_var("relative", relative))

    def tree(self):
        tree = ET.Element("rigid_bc", name=self.name, type="rigid_displacement")
        self.atr.fillTree(tree)
        return tree


class rigidRotation:
    _dof = Literal["Ru", "Rv", "Rw"]

    def __init__(
        self,
        name: str,
        material: str,
        dof: _dof,
        value: float = 1.0,
        relative: bool = False,
        attributes: dict | None = None,
    ):
        if attributes is None:
            attributes = {}

        self.name = name
        self.atr = sectionAttribs()
        self.atr.addAttrib(_var("rb", material))
        self.atr.addAttrib(_var("dof", dof))
        self.atr.addAttrib(_var("value", value, attributes))
        self.atr.addAttrib(_var("relative", relative))

    def tree(self):
        tree = ET.Element("rigid_bc", name=self.name, type="rigid_rotation")
        self.atr.fillTree(tree)
        return tree
