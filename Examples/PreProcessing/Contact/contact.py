import interFEBio.PreProcessing.Boundary as Boundary
import interFEBio.PreProcessing.Contact as Contact
import interFEBio.PreProcessing.LoadData as LoadData
import interFEBio.PreProcessing.Output as Output
import interFEBio.PreProcessing.Rigid as Rigid
from interFEBio.Mesh import Mesh
from interFEBio.PreProcessing import (
    Control,
    Material,
    Model,
    Step,
    material,
)

msh = Mesh.from_gmsh_msh("mesh/contactMesh.msh")

model = Model(modelfile="simpleContact.feb")

model.addMesh(msh)

model.addSurfacePair("contact_pair", "top_-z", "bottom_+z")

mat = Material(
    material(
        id=1,
        name="Material 1",
        type="incomp neo-Hookean",
        parameters={"density": 1, "k": 10.0, "G": 10.0},
        elementSet="bottom",
    )
)

rigidmat = Material(
    material(
        id=2,
        name="Material 2",
        type="rigid body",
        elementSet="top",
        parameters={"density": "1.0", "center_of_mass": "0,0,0"},
    )
)


model.addMaterial(mat)
model.addMaterial(rigidmat)
model.addGlobals()

fixBC = Boundary.fixedDisplacement("fix", "bottom_-z", x_dof=1, y_dof=1, z_dof=1)

loadController = LoadData.loadController(
    id=1, name="LC1", points=[[0, 0], [1, 1], [2, 0]]
)
rigidDisp = Rigid.rigidDisplacement(
    name="rigid displacement",
    material="Material 2",
    dof="z",
    value=-0.2,
    attributes={"value": {"lc": 1}},
)

rigidFix = Rigid.rigidFixed(
    name="rigidFix",
    material="Material 2",
    Rx_dof=1,
    Ry_dof=1,
    Ru_dof=1,
    Rv_dof=1,
    Rw_dof=1,
)

contact = Contact.sliding_elastic(
    "Contact",
    "contact_pair",
    penalty=100,
    two_pass=True,
    auto_penalty=True,
    flip_secondary=False,
    flip_primary=True,
)

solver = Control(time_steps=100, step_size=0.02, time_stepper=False)

step = Step(id=1, name="Step 1")
step.addBoundary(fixBC)
step.addRigid(rigidDisp)
step.addRigid(rigidFix)
step.addContact(contact)

step.addControl(solver)
model.addStep(step)
model.addLoadData(loadController)

model.addOutput(Output.plotVar("contact gap"))
model.addOutput(Output.plotVar("contact force"))
model.writeModel()
