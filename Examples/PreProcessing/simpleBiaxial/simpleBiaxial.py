import interFEBio.PreProcessing.Boundary as Boundary
import interFEBio.PreProcessing.LoadData as LoadData
from interFEBio.Mesh import Mesh
from interFEBio.PreProcessing import (
    Control,
    Material,
    Model,
    Step,
    material,
)

msh = Mesh.from_gmsh_msh("mesh/cube.msh")
model = Model(modelfile="simpleBiaxial.feb")
model.addMesh(msh)


mat = Material(
    material(
        id=1,
        name="Material 1",
        type="incomp neo-Hookean",
        parameters={"density": 1, "k": 10.0, "G": 10.0},
        elementSet="volume",
    )
)

model.addMaterial(mat)
model.addGlobals()

bcx = Boundary.fixedDisplacement("fix_x", "-x", x_dof=1)
bcy = Boundary.fixedDisplacement("fix_y", "-y", y_dof=1)
bcz = Boundary.fixedDisplacement("fix_z", "-z", z_dof=1)

loadController = LoadData.loadController(
    id=1, name="LC1", points=[[0, 0], [1, 1], [2, 0]]
)
dispx = Boundary.prescribedDisplacement(
    "Prescribed Displacement X", "+x", dof="x", value=1, lc=1
)
dispy = Boundary.prescribedDisplacement(
    "Prescribed Displacement Y", "+y", dof="y", value=1, lc=1
)

solver = Control(time_steps=20, step_size=0.1)

step = Step(id=1, name="Step 1")
step.addBoundary(dispx)
step.addBoundary(dispy)
step.addBoundary(bcx)
step.addBoundary(bcy)
step.addBoundary(bcz)
step.addControl(solver)

model.addStep(step)

model.addLoadData(loadController)
model.writeModel()
