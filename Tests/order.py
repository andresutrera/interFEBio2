# pv_osmesa_test.py
import os

os.environ["PYVISTA_OFF_SCREEN"] = "true"  # set before importing pyvista
os.environ["PYVISTA_USE_OSMESA"] = "true"  # tell pyvista to pick OSMesa
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import pyvista as pv

print(pv.Report())  # should show: Render Window : vtkOSOpenGLRenderWindow

mesh = pv.Sphere(phi_resolution=60, theta_resolution=60)
mesh["z"] = mesh.points[:, 2].astype("float32")

pl = pv.Plotter(off_screen=True, window_size=(800, 600))
pl.add_mesh(mesh, scalars="z", show_edges=False)
pl.camera_position = "iso"
pl.show(screenshot="pvtest.png")
print("wrote pvtest.png")
