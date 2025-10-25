from interFEBio.XPLT.XPLT import xplt
from interFEBio.Visualization.Plotter import PVBridge

import matplotlib.pyplot as plt

import pyvista as pv


# Load the XPLT file
xp = xplt("test.xplt")
xp.readAllStates()


print(xp.results.node["displacement"])


# bridge = PVBridge(xp.mesh)
# grid = bridge.domain_grid("Part1")

# grid2 = bridge.domain_grid("Part4")
# # # attach nodal vectors
# u = xp.results.node["displacement"]
# bridge.add_node_result(grid, u, t=-1, name="U")  # sets active vectors

# # attach element item tensors (symmetric 6 -> 9)
# s = xp.results.elem_item["stress"].region("Part1")
# bridge.add_elem_item_result(grid, s, domain="Part1", t=-1, name="S")

# pl = pv.Plotter()
# pl.add_mesh(grid, scalars="U", show_edges=True)
# pl.add_mesh(grid2, show_edges=True)
# pl.show()
