import numpy as np
import pyvista as pv
from interFEBio.XPLT.XPLT import xplt
from interFEBio.Visualization.Plotter import PVPlotter

import matplotlib.pyplot as plt

# Load the XPLT file
xp = xplt("ring.xplt")
xp.readAllStates()

print(xp.mesh.parts)

print(xp.dictionary)

print(xp.results)
stress = xp.results.node["displacement"]
print(stress)

stress = (
    xp.results.elem_item["stress"]
    .region("arteria")
    .time(":")
    .elems(0)
    .comp(slice(0, None))
)
print(stress)


force = (
    xp.results.face_region["contact force"].region("contactArtery").time(":").comp("y")
)
print(force)


plt.plot(xp.results.times(), force)
plt.show()
