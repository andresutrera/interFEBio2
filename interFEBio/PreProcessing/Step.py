"""
This module deals with boundary conditions
https://github.com/febiosoftware/FEBioStudio/blob/develop/FEMLib/FEBoundaryCondition.cpp

"""

import xml.etree.cElementTree as ET
from typing import TYPE_CHECKING, Protocol

from .Control import Control as _Control
from .dataClass import sectionAttribs

if TYPE_CHECKING:
    from .Boundary import fixedDisplacement, prescribedDisplacement
    from .Constraints import PrestrainGradient
    from .Contact import sliding_elastic
    from .Loads import surface_load
    from .Rigid import rigidDisplacement, rigidFixed, rigidRotation


class _TreeSerializable(Protocol):
    def tree(self) -> ET.Element: ...


if TYPE_CHECKING:
    BoundaryItem = fixedDisplacement | prescribedDisplacement | _TreeSerializable
    LoadItem = surface_load | _TreeSerializable
    RigidItem = rigidFixed | rigidDisplacement | rigidRotation | _TreeSerializable
    ContactItem = sliding_elastic | _TreeSerializable
    ConstraintItem = PrestrainGradient | _TreeSerializable
else:
    BoundaryItem = LoadItem = RigidItem = ContactItem = ConstraintItem = (
        _TreeSerializable
    )


class Step:
    def __init__(
        self,
        id: int = 1,
        name: str = "Step1",
    ):
        self.id = id
        self.name = name

        self.atr = sectionAttribs()

        self.bndatr: list[_TreeSerializable] = []
        self.loadsatr: list[_TreeSerializable] = []
        self.rigidatr: list[_TreeSerializable] = []
        self.contacatr: list[_TreeSerializable] = []
        self.constraintatr: list[_TreeSerializable] = []

    def addControl(self, ctrl: _Control | None = None) -> None:
        if ctrl is None:
            ctrl = _Control()
        self.atr.addAttrib(ctrl)

    def addBoundary(self, bound: BoundaryItem | None = None) -> None:
        if bound is not None:
            self.bndatr.append(bound)

    def addLoad(self, load: LoadItem | None = None) -> None:
        if load is not None:
            self.loadsatr.append(load)

    def addRigid(self, rigid: RigidItem | None = None) -> None:
        if rigid is not None:
            self.rigidatr.append(rigid)

    def addContact(self, cntc: ContactItem | None = None) -> None:
        if cntc is not None:
            self.contacatr.append(cntc)

    def addConstraint(self, cnstrnt: ConstraintItem | None = None) -> None:
        if cnstrnt is not None:
            self.constraintatr.append(cnstrnt)

    def tree(self) -> ET.Element:
        tree = ET.Element("step", id=str(self.id), name=self.name)
        self.atr.fillTree(tree)
        bndTree = ET.Element("Boundary")
        for item in self.bndatr:
            bndTree.append(item.tree())
        tree.append(bndTree)

        loadsTree = ET.Element("Loads")
        for item in self.loadsatr:
            loadsTree.append(item.tree())
        tree.append(loadsTree)

        rigidTree = ET.Element("Rigid")
        for item in self.rigidatr:
            rigidTree.append(item.tree())
        tree.append(rigidTree)

        contactTree = ET.Element("Contact")
        for item in self.contacatr:
            contactTree.append(item.tree())
        tree.append(contactTree)

        constraintTree = ET.Element("Constraints")
        for item in self.constraintatr:
            constraintTree.append(item.tree())
        tree.append(constraintTree)

        return tree
