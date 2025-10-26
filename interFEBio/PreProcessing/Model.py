"""
Module that defines and provides means for outputting the XML format .feb file.

Creating the object will initialize a skeleton of the model tree
"""

from __future__ import annotations

import xml.etree.cElementTree as ET
from typing import TYPE_CHECKING

from ..Mesh.Mesh import Mesh
from . import Control, Globals, LoadData, Output, Step

if TYPE_CHECKING:
    from .Control import Control as ControlBlock
    from .Globals import Constants as GlobalConstants
    from .LoadData import loadController
    from .Material import Material as MaterialSection
    from .Output import plotVar as PlotVar
    from .Step import Step as StepSection


class Model:
    """
    Create a model object.

    Args:
    ----------

        vers(str)       :   Spec version for the .feb output. Default = '3.0'
                            This library doesnt allow to write in different spec versions.

        modelfile(str)  :   Filename for the generated model file.

        encode(str)     :   Encoding for the .feb xml file. Default = "ISO-8859-1"

        steps(list)     :   list of dictionaries containing the steps of the analysis.
                            The format of each step is 'step name' : 'type'.
                                >>> steps=[{'Step01': 'solid'}, {'Step02': 'solid'}]
    """

    def __init__(self, vers="4.0", modelfile="default.feb", encode="ISO-8859-1"):
        """
        Constructor
        """
        self.modelfile = modelfile
        self.encode = encode
        self.mshObject: Mesh | None = None
        self.root = ET.Element("febio_spec", version=vers)
        self.firstModule = ET.SubElement(self.root, "Module", type="solid")
        self.globals = ET.SubElement(self.root, "Globals")
        self.material = ET.SubElement(self.root, "Material")
        self.mesh = ET.SubElement(self.root, "Mesh")
        self.meshDomains = ET.SubElement(self.root, "MeshDomains")
        self.meshData = ET.SubElement(self.root, "MeshData")
        self.Steps = ET.SubElement(self.root, "Step")
        self.loadData = ET.SubElement(self.root, "LoadData")
        self.output = ET.SubElement(self.root, "Output")
        self.plotVars = ET.SubElement(self.output, "plotfile", type="febio")

        self.addOutput(Output.plotVar("displacement"))
        self.addOutput(Output.plotVar("stress"))
        self.addOutput(Output.plotVar("relative volume"))

    def addGlobals(self, globals_obj: GlobalConstants | None = None) -> None:
        if globals_obj is None:
            globals_obj = Globals.Constants()
        self.globals.append(globals_obj.tree())

    def addControl(self, ctrl: ControlBlock | None = None) -> None:
        if ctrl is None:
            ctrl = Control.Control()
        self.root.insert(2, ctrl.tree())
        # self.control = ctrl.tree()

    def addMesh(self, msh: Mesh) -> None:
        self.mshObject = msh

        self.mesh.append(msh.to_feb_nodes_xml())  # Nodes

        for elset in msh.to_feb_elements_xml():  # Element domains
            self.mesh.append(elset)

        for surface in msh.to_feb_surfaces_xml():  # Surfaces
            self.mesh.append(surface)

        for nodeset in msh.to_feb_nodesets_xml():  # NodeSets
            self.mesh.append(nodeset)

        surface_pairs = getattr(msh, "surface_pairs", None)
        if surface_pairs is None:
            surface_pairs = getattr(msh, "surfacePairs", None)
        if surface_pairs:
            for name, pair in surface_pairs.items():
                surfPair = ET.Element("SurfacePair", name=name)
                ET.SubElement(surfPair, "primary").text = pair[0]
                ET.SubElement(surfPair, "secondary").text = pair[1]
                self.mesh.append(surfPair)

        mesh_data_entries = getattr(msh, "mesh_data", None)
        if mesh_data_entries is None:
            mesh_data_entries = getattr(msh, "meshData", None)
        if mesh_data_entries:
            for meshData in mesh_data_entries:
                root = meshData.getroot() if hasattr(meshData, "getroot") else None
                if root is not None:
                    self.meshData.append(root)
                elif isinstance(meshData, ET.Element):
                    self.meshData.append(meshData)

    def addStep(self, stp: StepSection | None = None) -> None:
        step_obj = stp if stp is not None else Step.Step()
        self.Steps.append(step_obj.tree())

    def addMaterial(
        self,
        material: MaterialSection | None = None,
        parameters: dict[str, object] | None = None,
    ) -> None:
        if material is None:
            raise ValueError("material must be provided")
        self.material.append(material.tree())
        if parameters is None:
            self.addMeshDomain(material)
        else:
            self.addMeshDomain(material, parameters=parameters)

    # def addMeshDomain(self, material:Material = None):
    #     type = self.mshObject.elsets[material.elementSet]['type']
    #     if type in ['quad4', 'tri3']:
    #         shellDom = ET.SubElement(self.meshDomains, 'ShellDomain', name=material.elementSet, mat=material.name)
    #         #ET.SubElement(shellDom, 'shell_normal_nodal').text = '1' ############ This was randomly deleted. Maybe is an error
    #     else:
    #         ET.SubElement(self.meshDomains, 'SolidDomain', name=material.elementSet, mat=material.name)

    def addMeshDomain(
        self,
        material: MaterialSection | None = None,
        _attributes: dict[str, object] | None = None,
        parameters: dict[str, object] | None = None,
    ) -> None:
        if material is None:
            raise ValueError("material must be provided")
        if self.mshObject is None:
            raise RuntimeError("Mesh must be added before defining mesh domains")

        base = material.baseMat
        element_set = getattr(base, "elementSet", None)
        if element_set is None:
            raise AttributeError("material.baseMat must define 'elementSet'")
        material_name = getattr(base, "name", None)
        if material_name is None:
            raise AttributeError("material.baseMat must define 'name'")

        try:
            domain = self.mshObject.getDomain(element_set)
        except KeyError as exc:
            raise KeyError(
                f"Unknown element set '{element_set}' for material '{material_name}'"
            ) from exc

        if len(domain) == 0:
            raise ValueError(f"Element set '{element_set}' has no elements assigned")

        domain_types = {str(t).lower() for t in domain.etype.tolist()}
        if len(domain_types) > 1:
            raise ValueError(
                f"Element set '{element_set}' mixes element types: {sorted(domain_types)}"
            )
        domain_type = next(iter(domain_types))

        if parameters is None:
            if domain_type in {"quad4", "tri3"}:
                ET.SubElement(
                    self.meshDomains,
                    "ShellDomain",
                    name=element_set,
                    mat=material_name,
                )
            else:
                ET.SubElement(
                    self.meshDomains,
                    "SolidDomain",
                    name=element_set,
                    mat=material_name,
                )
        else:
            if domain_type in {"line2", "line3"}:
                beamDomain = ET.SubElement(
                    self.meshDomains,
                    "BeamDomain",
                    name=element_set,
                    mat=material_name,
                    type="elastic-truss",
                )
                for prop, value in parameters.items():
                    ET.SubElement(beamDomain, prop).text = str(value)

    def addLoadData(self, load_data: loadController | None = None) -> None:
        controller = load_data if load_data is not None else LoadData.loadController()
        self.loadData.append(controller.tree())

    def addOutput(self, output: PlotVar | None = None) -> None:
        if output is None:
            raise ValueError("output must be provided")
        self.plotVars.append(output.tree())

    def writeModel(self) -> None:
        """
        Write the .feb model file

        """
        # Assemble XML tree
        tree = ET.ElementTree(self.root)

        # Make pretty format
        level = 0
        elem = tree.getroot()
        self.__indent(elem, level)

        self.remove_empty_elements(elem)
        # Write XML file
        tree.write(self.modelfile, encoding=self.encode)

    def __indent(self, elem: ET.Element, level: int) -> None:
        i = "\n" + level * "    "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "    "
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for child in elem:
                    self.__indent(child, level + 1)
                if not child.tail or not child.tail.strip():
                    child.tail = i
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def remove_empty_elements(self, element: ET.Element) -> None:
        for subelement in list(element):  # Use list() to safely iterate while modifying
            attrs = subelement.attrib
            # print("Attr: ",attrs)
            self.remove_empty_elements(subelement)
            if (
                not subelement.text and len(subelement) == 0 and not bool(attrs)
            ):  # No text and no children
                element.remove(subelement)
