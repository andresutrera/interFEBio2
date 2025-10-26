import numpy as np
import pyvista as pv


class BCPicker:
    """
    Interactively select nodes and faces, then assign displacement vectors.

    Shortcuts inside the window:
      1 = enable point picking (nodes)
      2 = enable face picking (cells on extracted surface)
      m = toggle target face set (A/B) for face picking
      r = clear selections and overlays
    Use the Python methods to assign boundary values:
      .apply_uniform(vec, scope="selected"|"all")
      .apply_linear(uA, uB)  # linear field from face A to face B through the volume
    """

    def __init__(self, grid, glyph_scale=0.02):
        self.grid = grid
        self.surface = grid.extract_surface().triangulate()
        self.pl = pv.Plotter()
        self.glyph_scale = glyph_scale

        # state
        self.selected_point_ids = set()
        self.face_mode = "A"  # which face set receives new picks
        self.faceA_cells = set()
        self.faceB_cells = set()
        self.faceA_mesh = None
        self.faceB_mesh = None
        self.point_markers = None
        self.glyph_actor = None

        # data field
        self.disp = np.zeros((grid.n_points, 3), dtype=float)
        self.grid.point_data["BC"] = self.disp

        # scene
        self.grid_actor = self.pl.add_mesh(
            self.grid, show_edges=True, scalars=None, opacity=1.0
        )
        self.surf_actor = self.pl.add_mesh(
            self.surface, color="lightgray", opacity=0.25, pickable=True
        )
        self.hud = self.pl.add_text(self._status_text(), font_size=10)

        # key bindings
        self.pl.add_key_event("1", self._enable_point_picking)
        self.pl.add_key_event("2", self._enable_cell_picking)
        self.pl.add_key_event("m", self._toggle_face_mode)
        self.pl.add_key_event("r", self._reset)

    # ---------- UI helpers ----------
    def _status_text(self):
        return (
            f"[1] pick nodes | [2] pick faces | [m] face={self.face_mode} | [r] reset\n"
            f"nodes: {len(self.selected_point_ids)} | faceA cells: {len(self.faceA_cells)} | faceB cells: {len(self.faceB_cells)}"
        )

    def _refresh_hud(self):
        self.hud.SetText(2, self._status_text())

    def _toggle_face_mode(self):
        self.face_mode = "B" if self.face_mode == "A" else "A"
        self._refresh_hud()

    def _reset(self):
        self.selected_point_ids.clear()
        self.faceA_cells.clear()
        self.faceB_cells.clear()
        if self.point_markers is not None:
            self.pl.remove_actor(self.point_markers, render=False)
            self.point_markers = None
        if self.faceA_mesh is not None:
            self.pl.remove_actor(self.faceA_mesh, render=False)
            self.faceA_mesh = None
        if self.faceB_mesh is not None:
            self.pl.remove_actor(self.faceB_mesh, render=False)
            self.faceB_mesh = None
        self._refresh_hud()
        self.pl.render()

    # ---------- picking ----------

    def _disable_pickers(self):
        # always clear previous mode
        self.pl.disable_picking()

    def _enable_point_picking(self):
        self._disable_pickers()

        def cb(point, picker):
            pid = int(self.grid.find_closest_point(point))
            self.selected_point_ids.add(pid)
            self._draw_point_markers()
            self._refresh_hud()
            self.pl.render()

        self.pl.enable_point_picking(
            callback=cb,
            picker="point",  # pick vertices
            use_picker=True,  # new API; replaces deprecated use_mesh
            left_clicking=True,
            show_message=False,
            show_point=True,
        )

    def _enable_cell_picking(self):
        self._disable_pickers()

        def cb(picked):
            import pyvista as pv

            ids = []
            if isinstance(picked, pv.MultiBlock):
                for blk in picked:
                    if blk is None:
                        continue
                    arr = blk.cell_data.get("vtkOriginalCellIds")
                    ids.extend(
                        (arr if arr is not None else np.arange(blk.n_cells)).tolist()
                    )
            else:
                if picked is None or picked.n_cells == 0:
                    return
                arr = picked.cell_data.get("vtkOriginalCellIds")
                ids = (arr if arr is not None else np.arange(picked.n_cells)).tolist()

            target = self.faceA_cells if self.face_mode == "A" else self.faceB_cells
            target.update(map(int, ids))
            self._draw_face_selection(which=self.face_mode)
            self._refresh_hud()
            self.pl.render()

        self.pl.enable_cell_picking(
            callback=cb,
            through=False,  # visible-only cells
            show=True,  # draw rectangle
            style="surface",
            start=True,  # picker active immediately
        )

    def _draw_point_markers(self):
        if not self.selected_point_ids:
            if self.point_markers is not None:
                self.pl.remove_actor(self.point_markers, render=False)
                self.point_markers = None
            return
        pts = self.grid.points[np.fromiter(self.selected_point_ids, dtype=int)]
        cloud = pv.PolyData(pts)
        if self.point_markers is not None:
            self.pl.remove_actor(self.point_markers, render=False)
        self.point_markers = self.pl.add_mesh(
            cloud, color="yellow", point_size=12, render_points_as_spheres=True
        )

    def _draw_face_selection(self, which="A"):
        cells = self.faceA_cells if which == "A" else self.faceB_cells
        actor_attr = "faceA_mesh" if which == "A" else "faceB_mesh"
        if not cells:
            if getattr(self, actor_attr) is not None:
                self.pl.remove_actor(getattr(self, actor_attr), render=False)
                setattr(self, actor_attr, None)
            return
        sel = self.surface.extract_cells(sorted(cells))
        color = "tomato" if which == "A" else "dodgerblue"
        # re-add actor each time to keep it simple and robust
        if getattr(self, actor_attr) is not None:
            self.pl.remove_actor(getattr(self, actor_attr), render=False)
        setattr(self, actor_attr, self.pl.add_mesh(sel, color=color, opacity=0.85))

    # ---------- assignment ----------
    def apply_uniform(self, vec, scope="selected"):
        vec = np.asarray(vec, float).reshape(1, 3)
        if scope == "all" or not self.selected_point_ids:
            idx = np.arange(self.grid.n_points, dtype=int)
        else:
            idx = np.fromiter(self.selected_point_ids, dtype=int)
        self.disp[idx, :] = vec
        self.grid.point_data["BC"] = self.disp
        self._update_glyphs()

    def apply_linear(self, uA, uB):
        """
        Build a linear field across the volume using two picked face patches.
        Axis is the vector from face-A centroid to face-B centroid.

        uA, uB: 3-vectors at face A and face B.
        """
        if not self.faceA_cells or not self.faceB_cells:
            raise RuntimeError("Pick both face A and face B before apply_linear().")

        A = self.surface.extract_cells(sorted(self.faceA_cells))
        B = self.surface.extract_cells(sorted(self.faceB_cells))

        pA = A.points.mean(axis=0)
        pB = B.points.mean(axis=0)
        v = pB - pA
        nv2 = float(np.dot(v, v))
        if nv2 == 0.0:
            raise RuntimeError("Face A and face B centroids coincide.")

        # projection coordinate s in [0,1] along v, clamped
        X = self.grid.points
        s = ((X - pA) @ v) / nv2
        s = np.clip(s, 0.0, 1.0)

        uA = np.asarray(uA, float).reshape(1, 3)
        uB = np.asarray(uB, float).reshape(1, 3)
        U = (1.0 - s)[:, None] * uA + s[:, None] * uB

        self.disp[:, :] = U
        self.grid.point_data["BC"] = self.disp
        self._update_glyphs()

    # ---------- viz of vectors ----------
    def _update_glyphs(self):
        # remove old glyphs
        if self.glyph_actor is not None:
            self.pl.remove_actor(self.glyph_actor, render=False)
            self.glyph_actor = None
        # add arrows for current BC field
        # avoid scale by magnitude so small vectors still visible
        glyphs = self.grid.glyph(orient="BC", factor=self.glyph_scale)
        self.glyph_actor = self.pl.add_mesh(glyphs)

        # also show magnitude as scalar on the grid for quick checks
        mag = np.linalg.norm(self.disp, axis=1)
        self.grid.point_data["|BC|"] = mag
        self.grid_actor.mapper.SetScalarModeToUsePointFieldData()
        self.grid_actor.mapper.SetArrayName("|BC|")
        self.pl.render()

    # ---------- run ----------
    def show(self):
        self.pl.show()


# ------------------ usage ------------------
# grid = <your pyvista.UnstructuredGrid or PolyData>
# gui = BCPicker(grid)
# gui.show()
# After you select nodes and/or faces interactively:
# gui.apply_uniform([0.0, 0.0, 1.0], scope="selected")   # or scope="all"
# gui.apply_linear(uA=[0,0,0], uB=[0,0,1])               # linear field A->B
