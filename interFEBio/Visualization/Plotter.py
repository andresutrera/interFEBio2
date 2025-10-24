# SafePVPlotter.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Literal

import os
import numpy as np
import pyvista as pv

from interFEBio.Mesh.Mesh import Mesh, SurfaceArray

# headless defaults
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
try:
    pv.rcParams["use_egl"] = True
except Exception:
    pass


def _try_start_xvfb() -> None:
    try:
        pv.start_xvfb()
    except Exception:
        pass


# ---------- base helpers ----------


def _to_zero_based(arr: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.int64, order="C")
    if a.size == 0:
        return a
    if a.min() >= 1 and a.max() <= n and 0 not in a:
        return a - 1
    return a


_DEFAULT_HEX8_PERM: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)


def _norm_label(raw: str, k: int) -> str:
    s = str(raw).upper()
    if s.startswith("FE_"):
        s = s[3:]
    if s in {"ELEM_HEX", "HEX", "HEXA", "BRICK"}:
        return "HEX20" if k >= 20 else "HEX8"
    if s in {"ELEM_TET", "TET"}:
        return "TET10" if k >= 10 else "TET4"
    if s in {"ELEM_PENTA", "WEDGE", "PENTA"}:
        return "QUADRATIC_WEDGE" if k >= 15 else "WEDGE"
    if s in {"ELEM_PYRA", "PYRA", "PYRAMID"}:
        return "PYRA5"
    if s in {"ELEM_QUAD", "QUAD"}:
        return "QUAD8" if k >= 8 else "QUAD4"
    if s in {"ELEM_TRI", "TRI"}:
        return "QUADRATIC_TRIANGLE" if k >= 6 else "TRI3"
    if s in {"ELEM_LINE", "LINE"}:
        return "LINE3" if k >= 3 else "LINE2"
    return s


def _is_2d_or_1d(vtk_id: int) -> bool:
    return vtk_id in (
        pv.CellType.QUAD,
        pv.CellType.QUADRATIC_QUAD,
        pv.CellType.TRIANGLE,
        pv.CellType.QUADRATIC_TRIANGLE,
        pv.CellType.LINE,
        pv.CellType.QUADRATIC_EDGE,
    )


def _is_2d(vtk_id: int) -> bool:
    return vtk_id in (
        pv.CellType.QUAD,
        pv.CellType.QUADRATIC_QUAD,
        pv.CellType.TRIANGLE,
        pv.CellType.QUADRATIC_TRIANGLE,
        pv.CellType.LINE,
        pv.CellType.QUADRATIC_EDGE,
    )


# ---------- geometry-based quad ordering ----------


def _order_quad_loop(ids4: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Return ids ordered as a non-self-intersecting loop around centroid."""
    P = xyz[ids4.astype(np.int64)]
    # local 2D frame via PCA
    Q = P - P.mean(axis=0)
    # SVD on 3xN
    U, _, _ = np.linalg.svd(Q.T, full_matrices=False)
    u, v = U[:, 0], U[:, 1]  # two principal directions in 3D
    x = Q @ u
    y = Q @ v
    ang = np.arctan2(y, x)
    order = np.argsort(ang)
    return ids4[order]


def _maybe_fix_quad_order(lbl: str, ids: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    if lbl != "QUAD4":
        return ids
    # If edges cross (order 0,1,3,2) the polygon will self-intersect in projection.
    # Use angle sort around centroid to force a proper loop.
    return _order_quad_loop(ids, xyz)


# ---------- von Mises ----------


def von_mises(sig: np.ndarray) -> np.ndarray:
    sxx, syy, szz, syz, sxz, sxy = (sig[..., i] for i in range(6))
    j2 = (
        (sxx - syy) ** 2
        + (syy - szz) ** 2
        + (szz - sxx) ** 2
        + 6.0 * (sxy * sxy + sxz * sxz + syz * syz)
    ) / 6.0
    return np.sqrt(np.maximum(j2, 0.0)).astype(np.float32)


# ---------- cache ----------


@dataclass
class _Cache:
    point_grid: pv.DataSet | None = None
    elem_index_map: np.ndarray | None = None
    cell_grid_by_domain: Dict[str, pv.UnstructuredGrid] = field(default_factory=dict)
    cell_ids_by_domain: Dict[str, np.ndarray] = field(default_factory=dict)
    domain_elem_mask: Dict[str, np.ndarray] = field(default_factory=dict)
    surface_poly_by_name: Dict[str, pv.PolyData] = field(default_factory=dict)


# ---------- plotter ----------


class PVPlotter:
    def __init__(
        self, offscreen: bool = True, window_size: Tuple[int, int] = (900, 700)
    ) -> None:
        self._mesh: Mesh | None = None
        self._xplt: object | None = None
        self._t: int = 0
        self._cache = _Cache()
        self._hex8_perm_by_domain: Dict[str, Tuple[int, ...]] = {}
        self._elem_domain_of: List[str] | None = None
        self._eid2row: Dict[int, int] | None = None

        if offscreen:
            _try_start_xvfb()
        self.pl = pv.Plotter(off_screen=offscreen, window_size=window_size)
        self._offscreen = bool(self.pl.off_screen)
        self._allow_interactive = not self._offscreen
        try:
            import vtk

            vtk.vtkObject.GlobalWarningDisplayOff()
        except Exception:
            pass

    def __enter__(self) -> "PVPlotter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finalize()

    # ---- element-id mapping ----

    def _elem_id_array(self) -> np.ndarray | None:
        if self._mesh is None:
            return None
        for attr in ("ids", "elem_ids", "id"):
            if hasattr(self._mesh.elements, attr):
                a = np.asarray(getattr(self._mesh.elements, attr))
                if a.size:
                    return a.astype(np.int64, copy=False)
        return None

    def _ensure_eid_map(self) -> None:
        if self._eid2row is not None:
            return
        eid = self._elem_id_array()
        self._eid2row = (
            {int(eid[i]): i for i in range(eid.shape[0])} if eid is not None else {}
        )

    def _map_elem_refs_to_rows(self, refs: np.ndarray) -> np.ndarray:
        if self._mesh is None:
            return refs
        E = self._mesh.elements.conn.shape[0]
        r = np.asarray(refs, dtype=np.int64)
        if r.size == 0:
            return r
        if r.min() >= 0 and r.max() < E:
            return r  # already 0-based rows
        if r.min() >= 1 and r.max() <= E and 0 not in r:
            return r - 1  # 1-based rows
        self._ensure_eid_map()
        if not self._eid2row:
            raise ValueError("Element ids present but elements.ids not available")
        out = np.empty_like(r)
        for i, v in enumerate(r):
            if int(v) not in self._eid2row:
                raise ValueError(f"Unknown element id in domain: {int(v)}")
            out[i] = self._eid2row[int(v)]
        return out

    # ---- attach ----

    def _build_elem_domain_map(self) -> None:
        if self._mesh is None:
            self._elem_domain_of = None
            return
        E = self._mesh.elements.conn.shape[0]
        dom_of = np.full(E, "", dtype=object)
        for name, idx in self._mesh.parts.items():
            ids = self._map_elem_refs_to_rows(np.asarray(idx, dtype=np.int64))
            dom_of[ids] = name
        self._elem_domain_of = dom_of.tolist()

    def attach_mesh(self, mesh: Mesh) -> None:
        self._mesh = mesh
        self._xplt = None
        self._t = 0
        self._cache = _Cache()
        self._hex8_perm_by_domain.clear()
        self._eid2row = None
        self._build_elem_domain_map()

    def attach_xplt(self, xp: object) -> None:
        self._xplt = xp
        self._mesh = getattr(xp, "mesh")
        self._t = 0
        self._cache = _Cache()
        self._hex8_perm_by_domain.clear()
        self._eid2row = None
        self._build_elem_domain_map()

    # ---- HEX8 perm calibration ----

    @staticmethod
    def _hex8_candidates() -> List[Tuple[int, ...]]:
        out = []
        for base in [(0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 3, 2, 4, 5, 7, 6)]:
            for r in range(4):
                b = [base[(i + r) % 4] for i in range(4)]
                t = [base[4 + ((i + r) % 4)] for i in range(4)]
                out.append(tuple(b + t))
        return out

    @staticmethod
    def _hex_faces_ids(ids8p: Sequence[int]) -> List[List[int]]:
        i = list(ids8p)
        return [
            [i[0], i[1], i[2], i[3]],
            [i[4], i[5], i[6], i[7]],
            [i[0], i[1], i[5], i[4]],
            [i[1], i[2], i[6], i[5]],
            [i[2], i[3], i[7], i[6]],
            [i[3], i[0], i[4], i[7]],
        ]

    @staticmethod
    def _rot_eq4(a: List[int], b: List[int]) -> bool:
        for s in range(4):
            if all(a[(k + s) % 4] == b[k] for k in range(4)):
                return True
        return False

    def calibrate_hex8_permutation(
        self, max_hex_samples: int = 3000
    ) -> Dict[str, Tuple[int, ...]]:
        if self._mesh is None:
            raise RuntimeError("Mesh not attached")
        m = self._mesh
        EA = m.elements
        xyz = np.asarray(m.nodes.xyz)
        N = xyz.shape[0]
        conn0 = _to_zero_based(EA.conn, N)

        quad_list: List[List[int]] = []
        for sa in m.surfaces.values():
            F = _to_zero_based(np.asarray(sa.faces, dtype=np.int64), N)
            K = np.asarray(sa.nper, dtype=np.int64)
            for i in np.nonzero(K == 4)[0]:
                # ensure proper loop for matching
                quad_list.append(_order_quad_loop(F[i, :4], xyz).tolist())
        if not quad_list:
            self._hex8_perm_by_domain = {name: _DEFAULT_HEX8_PERM for name in m.parts}
            return dict(self._hex8_perm_by_domain)

        quad_sets = {frozenset(q) for q in quad_list}
        et = np.char.upper(EA.etype.astype(str))
        is_hex = np.isin(et, ["ELEM_HEX", "HEX", "HEXA", "BRICK"])
        perms = self._hex8_candidates()
        result: Dict[str, Tuple[int, ...]] = {}

        for dname, eidx in m.parts.items():
            idsE = self._map_elem_refs_to_rows(np.asarray(eidx, dtype=np.int64))
            dom_hex = idsE[is_hex[idsE]]
            if dom_hex.size == 0:
                continue
            sample: List[int] = []
            for e in dom_hex:
                ids8 = conn0[e, :8]
                cand = [
                    frozenset(ids8[[0, 1, 2, 3]]),
                    frozenset(ids8[[4, 5, 6, 7]]),
                    frozenset(ids8[[0, 1, 5, 4]]),
                    frozenset(ids8[[1, 2, 6, 5]]),
                    frozenset(ids8[[2, 3, 7, 6]]),
                    frozenset(ids8[[3, 0, 4, 7]]),
                ]
                if any(c in quad_sets for c in cand):
                    sample.append(int(e))
                if len(sample) >= max_hex_samples:
                    break
            if not sample:
                result[dname] = _DEFAULT_HEX8_PERM
                continue

            scores = np.zeros(len(perms), dtype=np.int64)
            for pi, perm in enumerate(perms):
                sc = 0
                p = np.array(perm, dtype=np.int64)
                for e in sample:
                    ids8p = conn0[e, p]
                    for f in self._hex_faces_ids(ids8p):
                        fset = frozenset(f)
                        if fset in quad_sets:
                            if any(
                                self._rot_eq4(f, q)
                                for q in quad_list
                                if frozenset(q) == fset
                            ):
                                sc += 1
                scores[pi] = sc
            result[dname] = perms[int(np.argmax(scores))]

        self._hex8_perm_by_domain = result
        return dict(result)

    # ---- pulls ----

    def times(self) -> np.ndarray:
        if self._xplt is None:
            return np.asarray([], dtype=float)
        return getattr(self._xplt, "results").times()

    def set_time_index(self, i: int) -> None:
        if self._xplt is None:
            self._t = 0
            return
        nt = len(getattr(self._xplt, "results"))
        self._t = int(np.clip(i, 0, max(nt - 1, 0)))

    def _nodal(self, name: str) -> np.ndarray:
        r = getattr(self._xplt, "results")
        return np.asarray(r[name][self._t, :, :])

    def _elem(self, name: str, domain: str) -> np.ndarray:
        r = getattr(self._xplt, "results")
        view = r[name].domain(domain)
        return np.asarray(view[self._t, :, :])

    def _face(self, name: str, surface: str) -> np.ndarray:
        r = getattr(self._xplt, "results")
        view = r[name].surface(surface)
        return np.asarray(view[self._t, :, :])

    # ---- VTK build ----

    def add_solid_surface(self, color=None, opacity=0.9, style="surface"):
        grid = self._build_point_grid()
        if isinstance(grid, pv.UnstructuredGrid):
            surf = grid.extract_surface().clean().compute_normals()
            return self.pl.add_mesh(
                surf,
                style=style,
                color=color,
                opacity=opacity,
                smooth_shading=False,
                culling="back",
            )
        return self.pl.add_mesh(grid, style=style, color=color, opacity=opacity)

    def _build_point_grid(self) -> pv.DataSet:
        if self._cache.point_grid is not None:
            return self._cache.point_grid
        if self._mesh is None:
            raise RuntimeError("Mesh not attached")

        m = self._mesh
        xyz = np.ascontiguousarray(np.asarray(m.nodes.xyz, dtype=np.float64))
        N = xyz.shape[0]
        conn = _to_zero_based(np.asarray(m.elements.conn, dtype=np.int64), N)
        nper = np.asarray(m.elements.nper, dtype=np.int64)
        etype = np.asarray(m.elements.etype, dtype=object)

        map_vtk: dict[str, int] = {
            "TET4": pv.CellType.TETRA,
            "TET10": pv.CellType.QUADRATIC_TETRA,
            "HEX8": pv.CellType.HEXAHEDRON,
            "HEX20": pv.CellType.QUADRATIC_HEXAHEDRON,
            "WEDGE": pv.CellType.WEDGE,
            "QUADRATIC_WEDGE": pv.CellType.QUADRATIC_WEDGE,
            "PYRA5": pv.CellType.PYRAMID,
            "QUAD4": pv.CellType.QUAD,
            "QUAD8": pv.CellType.QUADRATIC_QUAD,
            "TRI3": pv.CellType.TRIANGLE,
            "QUADRATIC_TRIANGLE": pv.CellType.QUADRATIC_TRIANGLE,
            "LINE2": pv.CellType.LINE,
            "LINE3": pv.CellType.QUADRATIC_EDGE,
        }

        dom_of = self._elem_domain_of if self._elem_domain_of is not None else []

        E = conn.shape[0]
        pieces3d: List[np.ndarray] = []
        types3d: List[int] = []
        pieces2d: List[np.ndarray] = []
        types2d: List[int] = []
        index_map = np.full(E, -1, dtype=np.int64)

        for e in range(E):
            k = int(nper[e]) if e < nper.size else 0
            if k <= 0:
                continue
            lbl = _norm_label(etype[e], k)
            vtk_id = map_vtk.get(lbl)
            if vtk_id is None:
                continue

            ids = conn[e, :k]

            # apply per-domain HEX8 permutation
            if lbl == "HEX8":
                dname = dom_of[e] if e < len(dom_of) else ""
                perm = self._hex8_perm_by_domain.get(dname, _DEFAULT_HEX8_PERM)
                ids = ids[list(perm)]

            # fix QUAD4 ordering to avoid self-crossing wireframe
            if lbl == "QUAD4":
                ids = _maybe_fix_quad_order(lbl, ids, xyz)

            if ids.min() < 0 or ids.max() >= N:
                continue

            rec = np.concatenate(([k], ids.astype(np.int64)))
            if _is_2d(vtk_id):
                pieces2d.append(rec)
                types2d.append(vtk_id)
            else:
                index_map[e] = len(types3d)
                pieces3d.append(rec)
                types3d.append(vtk_id)

        if types3d:
            cells3d = np.ascontiguousarray(np.concatenate(pieces3d).astype(np.int64))
            ctype3d = np.ascontiguousarray(np.array(types3d, dtype=np.uint8))
            grid3d = pv.UnstructuredGrid(cells3d, ctype3d, xyz).clean(tolerance=0.0)
            self._cache.point_grid = grid3d
        else:
            self._cache.point_grid = pv.PolyData(xyz)

        if types2d:
            cells2d = np.ascontiguousarray(np.concatenate(pieces2d).astype(np.int64))
            ctype2d = np.ascontiguousarray(np.array(types2d, dtype=np.uint8))
            shell_grid = pv.UnstructuredGrid(cells2d, ctype2d, xyz)
            self._cache.surface_poly_by_name["__shells__"] = (
                shell_grid.extract_geometry().clean()
            )
        else:
            self._cache.surface_poly_by_name.pop("__shells__", None)

        self._cache.elem_index_map = index_map
        return self._cache.point_grid

    def _build_cell_grid_for_domain(self, domain: str) -> pv.UnstructuredGrid:
        if domain in self._cache.cell_grid_by_domain:
            return self._cache.cell_grid_by_domain[domain]
        base = self._build_point_grid()
        if not isinstance(base, pv.UnstructuredGrid):
            raise ValueError("Mesh has no volume elements; domain view invalid")
        refs = np.asarray(self._mesh.parts[domain], dtype=np.int64)
        ids = self._map_elem_refs_to_rows(refs)
        index_map = self._cache.elem_index_map
        if index_map is None:
            raise RuntimeError("Element index map missing")
        mapped = index_map[ids]
        valid_mask = mapped >= 0
        if not np.any(valid_mask):
            raise ValueError(f"Domain '{domain}' has no renderable volume elements")
        valid_cells = np.ascontiguousarray(mapped[valid_mask], dtype=np.int64)
        sub = base.extract_cells(valid_cells)
        self._cache.cell_grid_by_domain[domain] = sub
        self._cache.cell_ids_by_domain[domain] = valid_cells
        self._cache.domain_elem_mask[domain] = valid_mask
        return sub

    def _build_surface_poly(self, surface: str) -> pv.PolyData:
        if surface in self._cache.surface_poly_by_name:
            obj = self._cache.surface_poly_by_name[surface]
            return (
                obj if isinstance(obj, pv.PolyData) else obj.extract_geometry().clean()
            )
        if self._mesh is None:
            raise RuntimeError("Mesh not attached")
        sa: SurfaceArray = self._mesh.surfaces[surface]
        N = self._mesh.nodes.xyz.shape[0]
        faces = _to_zero_based(np.asarray(sa.faces, dtype=np.int64), N)
        nper = np.asarray(sa.nper, dtype=np.int64)
        if faces.size == 0:
            poly = pv.PolyData(self._mesh.nodes.xyz.astype(np.float64))
            self._cache.surface_poly_by_name[surface] = poly
            return poly
        pcs: List[np.ndarray] = []
        for f in range(faces.shape[0]):
            k = int(nper[f])
            if k <= 0:
                continue
            ids = faces[f, :k].astype(np.int64)
            if k == 4:  # enforce proper loop
                ids = _order_quad_loop(ids, self._mesh.nodes.xyz.astype(np.float64))
            pcs.append(np.array([k], dtype=np.int64))
            pcs.append(ids)
        cells = np.concatenate(pcs) if pcs else np.array([], dtype=np.int64)
        poly = pv.PolyData(self._mesh.nodes.xyz.astype(np.float64), cells).clean()
        self._cache.surface_poly_by_name[surface] = poly
        return poly

    # ---- set/add ----

    def _set_point_scalars(self, key: str, arr: np.ndarray) -> pv.DataSet:
        grid = self._build_point_grid()
        vec = np.asarray(arr, dtype=np.float32, order="C")
        if key in grid.point_data and grid.point_data[key].shape == vec.shape:
            grid.point_data[key][:] = vec
        else:
            grid.point_data[key] = vec
        return grid

    def _set_cell_scalars(
        self, grid: pv.UnstructuredGrid, key: str, arr: np.ndarray
    ) -> None:
        vec = np.asarray(arr, dtype=np.float32, order="C")
        if key in grid.cell_data and grid.cell_data[key].shape == vec.shape:
            grid.cell_data[key][:] = vec
        else:
            grid.cell_data[key] = vec

    def add_mesh(
        self,
        style: Literal["surface", "wireframe"] = "wireframe",
        color: str | Sequence[float] | None = None,
        opacity: float = 1.0,
    ):
        grid = self._build_point_grid()
        return self.pl.add_mesh(
            grid,
            style=style,
            color=color,
            opacity=opacity,
            smooth_shading=False,
            culling="back",
        )

    def add_shells(
        self,
        style: Literal["surface", "wireframe"] = "wireframe",
        color: str | Sequence[float] | None = None,
        opacity: float = 0.3,
    ):
        _ = self._build_point_grid()
        shells = self._cache.surface_poly_by_name.get("__shells__")
        if shells is None:
            return None
        return self.pl.add_mesh(shells, style=style, color=color, opacity=opacity)

    def add_surface(
        self,
        surface: str,
        style: Literal["surface", "wireframe"] = "surface",
        color: str | Sequence[float] | None = None,
        opacity: float = 1.0,
    ):
        poly = self._build_surface_poly(surface)
        return self.pl.add_mesh(poly, style=style, color=color, opacity=opacity)

    def add_scalar_nodal(
        self,
        name: str,
        comp: int | str | None = None,
        cmap: str | None = None,
        clim: Tuple[float, float] | None = None,
        scalar_bar: bool = True,
        alias: str | None = None,
    ):
        arr = self._nodal(name)
        if isinstance(comp, int):
            scal = arr[:, comp]
            key = alias or f"{name}_{comp}"
        elif isinstance(comp, str) and comp.lower() == "magnitude":
            scal = np.linalg.norm(arr, axis=1)
            key = alias or f"{name}_mag"
        else:
            scal = arr[:, 0] if arr.ndim == 2 and arr.shape[1] > 1 else np.ravel(arr)
            key = alias or (name if scal.ndim == 1 else f"{name}_0")
        grid = self._set_point_scalars(key, scal)
        return self.pl.add_mesh(
            grid, scalars=key, cmap=cmap, clim=clim, show_scalar_bar=scalar_bar
        )

    def add_scalar_elem(
        self,
        name: str,
        domain: str,
        comp: int | str | None = None,
        cmap: str | None = None,
        clim: Tuple[float, float] | None = None,
        scalar_bar: bool = True,
        alias: str | None = None,
    ):
        arr = self._elem(name, domain)
        grid = self._build_cell_grid_for_domain(domain)
        mask = self._cache.domain_elem_mask.get(domain)
        if mask is not None and mask.shape[0] == arr.shape[0]:
            arr = arr[mask]
        if arr.shape[0] != grid.n_cells:
            raise ValueError(
                f"Element result '{name}' rows {arr.shape[0]} != domain '{domain}' cells {grid.n_cells}"
            )
        if isinstance(comp, int):
            scal = arr[:, comp]
            key = alias or f"{name}_{comp}"
        elif isinstance(comp, str) and comp.lower() in ("vonmises", "von_mises", "vm"):
            scal = von_mises(arr)
            key = alias or f"{name}_vm"
        else:
            scal = arr[:, 0] if arr.ndim == 2 and arr.shape[1] > 1 else np.ravel(arr)
            key = alias or (name if scal.ndim == 1 else f"{name}_0")
        self._set_cell_scalars(grid, key, scal)
        return self.pl.add_mesh(
            grid, scalars=key, cmap=cmap, clim=clim, show_scalar_bar=scalar_bar
        )

    def add_scalar_face(
        self,
        name: str,
        surface: str,
        comp: int | None = None,
        cmap: str | None = None,
        clim: Tuple[float, float] | None = None,
        scalar_bar: bool = True,
        alias: str | None = None,
    ):
        arr = self._face(name, surface)
        poly = self._build_surface_poly(surface)
        scal = (
            arr[:, comp]
            if isinstance(comp, int)
            else (arr[:, 0] if arr.shape[1] > 1 else np.ravel(arr))
        )
        key = alias or (f"{name}_{comp}" if isinstance(comp, int) else name)
        vec = np.asarray(scal, dtype=np.float32, order="C")
        if key in poly.cell_data and poly.cell_data[key].shape == vec.shape:
            poly.cell_data[key][:] = vec
        else:
            poly.cell_data[key] = vec
        return self.pl.add_mesh(
            poly, scalars=key, cmap=cmap, clim=clim, show_scalar_bar=scalar_bar
        )

    def add_vectors_nodal(
        self,
        name: str,
        scale: float = 1.0,
        glyph: bool = True,
        stride: int = 1,
        alias: str | None = None,
    ):
        arr = self._nodal(name)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("Vector field must have at least 3 components")
        vec = arr[:, :3].astype(np.float32, order="C")
        key = alias or f"{name}_vec"
        grid = self._set_point_scalars(key, vec)
        if glyph:
            pts = grid.points[::stride]
            vv = vec[::stride]
            mask = np.isfinite(vv).all(axis=1)
            return self.pl.add_arrows(pts[mask], vv[mask], mag=scale)
        warped = grid.warp_by_vector(vectors=key, factor=scale)
        return self.pl.add_mesh(warped)

    # ---- nodes only ----
    def add_nodes(
        self,
        point_size: int = 8,
        color: str | Sequence[float] | None = None,
        as_spheres: bool = True,
    ):
        if self._mesh is None:
            raise RuntimeError("Mesh not attached")
        xyz = np.asarray(self._mesh.nodes.xyz, dtype=np.float64)
        pts = pv.PolyData(xyz)
        return self.pl.add_mesh(
            pts, render_points_as_spheres=as_spheres, point_size=point_size, color=color
        )

    # ---- view / IO ----

    def view_isometric(self) -> None:
        self.pl.view_isometric()

    def set_background(self, color: str | Sequence[float] = "white") -> None:
        self.pl.set_background(color)

    def show(self, interactive: bool | None = None, auto_close: bool = True):
        inter = (
            self._allow_interactive
            if interactive is None
            else bool(interactive and self._allow_interactive)
        )
        return self.pl.show(interactive=inter, auto_close=auto_close)

    def screenshot(self, path: str) -> None:
        if self._offscreen:
            self.pl.render()
            self.pl.screenshot(path)
        else:
            self.pl.show(interactive=False, auto_close=False)
            self.pl.screenshot(path)

    def clear(self) -> None:
        self.pl.clear()

    def finalize(self) -> None:
        try:
            self.pl.close()
        except Exception:
            pass
        try:
            pv.close_all()
        except Exception:
            pass
        import gc

        gc.collect()

    # ---- info ----
    def available_fields(self) -> Dict[str, List[str]]:
        if self._xplt is None:
            return {"nodal": [], "elem": [], "face": []}
        r = getattr(self._xplt, "results")
        return {
            "nodal": list(getattr(r, "_nodal", {}).keys()),
            "elem": list(getattr(r, "_elem", {}).keys()),
            "face": list(getattr(r, "_face", {}).keys()),
        }

    def domains(self) -> List[str]:
        if self._mesh is None:
            return []
        return list(self._mesh.parts.keys())

    def surfaces(self) -> List[str]:
        if self._mesh is None:
            return []
        return list(self._mesh.surfaces.keys())

    def add_domain(
        self,
        domain: str,
        style: Literal["surface", "wireframe", "surface_lines"] = "wireframe",
        color: str | Sequence[float] | None = None,
        opacity: float = 1.0,
        edge_color: str = "black",  # Default edge color for surface_lines style
        line_width: float = 1.0,  # Default line width for edges
    ):
        """Plot all elements that belong to a single domain with the requested style."""

        # Build domain-specific datasets (volume grid + surface)
        vol_grid, geom_poly = self._build_domain_datasets(domain)

        actors = []

        # Handle volume grid (3D elements)
        if vol_grid is not None:
            actors.append(
                self.pl.add_mesh(
                    vol_grid,
                    style=style,
                    color=color,
                    opacity=opacity,
                    smooth_shading=False,
                    culling="back",
                )
            )

        # Handle surface geometry (2D or 1D elements)
        if geom_poly is not None:
            if style == "surface_lines":
                # Extract surface and show element edges
                geom_poly_with_edges = geom_poly.copy()
                actors.append(
                    self.pl.add_mesh(
                        geom_poly_with_edges,
                        style="surface",
                        color=color,
                        opacity=opacity,
                        smooth_shading=False,
                        culling="back",
                        show_edges=True,  # Display element edges
                        edge_color=edge_color,
                        line_width=line_width,
                    )
                )
            else:
                actors.append(
                    self.pl.add_mesh(
                        geom_poly,
                        style=style,
                        color=color,
                        opacity=opacity,
                        smooth_shading=False,
                        culling="back",
                    )
                )

        return actors

    def _build_domain_datasets(self, domain: str):
        if self._mesh is None:
            raise RuntimeError("Mesh not attached")
        m = self._mesh
        xyz = np.ascontiguousarray(np.asarray(m.nodes.xyz, dtype=np.float64))
        N = xyz.shape[0]
        conn = np.asarray(m.elements.conn, dtype=np.int64)
        nper = np.asarray(m.elements.nper, dtype=np.int64)
        etype = np.asarray(m.elements.etype, dtype=object)
        idsE = self._map_elem_refs_to_rows(np.asarray(m.parts[domain], dtype=np.int64))

        map_vtk = {
            "TET4": pv.CellType.TETRA,
            "TET10": pv.CellType.QUADRATIC_TETRA,
            "HEX8": pv.CellType.HEXAHEDRON,
            "HEX20": pv.CellType.QUADRATIC_HEXAHEDRON,
            "WEDGE": pv.CellType.WEDGE,
            "QUADRATIC_WEDGE": pv.CellType.QUADRATIC_WEDGE,
            "PYRA5": pv.CellType.PYRAMID,
            "QUAD4": pv.CellType.QUAD,
            "QUAD8": pv.CellType.QUADRATIC_QUAD,
            "TRI3": pv.CellType.TRIANGLE,
            "QUADRATIC_TRIANGLE": pv.CellType.QUADRATIC_TRIANGLE,
            "LINE2": pv.CellType.LINE,
            "LINE3": pv.CellType.QUADRATIC_EDGE,
        }

        pieces3d: List[np.ndarray] = []
        types3d: List[int] = []
        pieces2d1d: List[np.ndarray] = []
        types2d1d: List[int] = []

        for e in idsE:
            k = int(nper[e])
            if k <= 0:
                continue
            lbl = _norm_label(etype[e], k)
            vtk_id = map_vtk.get(lbl)
            if vtk_id is None:
                continue
            ids = conn[e, :k]
            if ids.min() < 0 or ids.max() >= N:
                continue
            rec = np.concatenate(([k], ids.astype(np.int64)))
            if _is_2d_or_1d(vtk_id):
                pieces2d1d.append(rec)
                types2d1d.append(vtk_id)
            else:
                pieces3d.append(rec)
                types3d.append(vtk_id)

        vol_grid = None
        geom_poly = None
        if types3d:
            cells3d = np.ascontiguousarray(np.concatenate(pieces3d).astype(np.int64))
            ctype3d = np.ascontiguousarray(np.array(types3d, dtype=np.uint8))
            vol_grid = pv.UnstructuredGrid(cells3d, ctype3d, xyz).clean(tolerance=0.0)
        if types2d1d:
            cells = np.ascontiguousarray(np.concatenate(pieces2d1d).astype(np.int64))
            ctypes = np.ascontiguousarray(np.array(types2d1d, dtype=np.uint8))
            shell_line = pv.UnstructuredGrid(cells, ctypes, xyz)
            geom_poly = shell_line.extract_geometry().clean()

        return vol_grid, geom_poly
