import numpy as np

import vtk, pyvista as pv


from interFEBio.XPLT.XPLT import xplt
from interFEBio.Visualization.Plotter import PVPlotter

print("supports_plotting:", pv.system_supports_plotting())

xp = xplt("ring.xplt")

xp.readAllStates()
print(xp.mesh.parts)

pvplt = PVPlotter(offscreen=False)  # or True if headless
pvplt.attach_xplt(xp)


# print(pvplt.domains())  # e.g. ['arteria', 'contactPin', 'wires']

pvplt.add_domain("arteria", style="surface", color="lightgray", opacity=1.0)
pvplt.add_domain("part_1", style="surface", color="black", opacity=1.0)

# # 4) View
pvplt.view_isometric()
pvplt.show()
