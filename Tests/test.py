from interFEBio.XPLT import xplt
from interFEBio.Visualization.Plotter import PVBridge

import matplotlib.pyplot as plt

import pyvista as pv


# Load the XPLT file
xp = xplt("microModel.xplt")
xp.readAllStates()
print(xp.mesh.parts)

# ns1 = xp.mesh.nodesets["Part1"]

bridge = PVBridge(xp.mesh)
grid = bridge.domain_grid("ground")


# grid2 = bridge.domain_grid("Part4")
U_view = xp.results.node["displacement"]
U = U_view[-1, ":", 0]  # (N,3) at time 0
bridge.add_node_result_array(grid, view=U_view, data=U, name="U")  # vectors

# attach element item tensors (symmetric 6 -> 9)
s = xp.results.elem_item["stress"]
# bridge.add_elem_item_array(
#     grid, view=s, domain="Part1", data=s.region("Part1").time(0).comp(0), name="S"
# )

pl = pv.Plotter()
pl.add_mesh(grid, scalars="U", show_edges=True)
# pl.add_mesh(grid2, show_edges=True)
pl.show()
