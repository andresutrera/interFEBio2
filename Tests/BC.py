from interFEBio.XPLT.XPLT import xplt
from interFEBio.Visualization.Plotter import PVBridge
from interFEBio.Visualization.BCPicker import BCPicker
import matplotlib.pyplot as plt

import pyvista as pv


# Load the XPLT file
xp = xplt("test.xplt")
xp.readAllStates()
bridge = PVBridge(xp.mesh)
grid = bridge.domain_grid("Part1")

gui = BCPicker(grid)
gui.show()
