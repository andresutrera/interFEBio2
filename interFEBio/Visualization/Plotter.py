from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pyvista as pv
from interFEBio.Mesh.Mesh import Mesh, SurfaceArray
from interFEBio.XPLT.Enums import FEDataType
from interFEBio.XPLT.XPLT import (
    ItemResultView,
    MultResultView,
    NodeResultView,
    RegionResultView,
)  # adjust import path

# -------------------------- internal helpers --------------------------

_VTK_MAP: Dict[tuple[str, int], int] = {
    ("TET4", 4): pv.CellType.TETRA,
    ("TET10", 10): pv.CellType.QUADRATIC_TETRA,
    ("HEX8", 8): pv.CellType.HEXAHEDRON,
    ("HEX20", 20): pv.CellType.QUADRATIC_HEXAHEDRON,
    ("WEDGE", 6): pv.CellType.WEDGE,
    ("QUADRATIC_WEDGE", 15): pv.CellType.QUADRATIC_WEDGE,
    ("PYRA5", 5): pv.CellType.PYRAMID,
    ("QUAD4", 4): pv.CellType.QUAD,
    ("QUAD8", 8): pv.CellType.QUADRATIC_QUAD,
    ("TRI3", 3): pv.CellType.TRIANGLE,
    ("QUADRATIC_TRIANGLE", 6): pv.CellType.QUADRATIC_TRIANGLE,
    ("LINE2", 2): pv.CellType.LINE,
    ("LINE3", 3): pv.CellType.QUADRATIC_EDGE,
}


def _norm_label(raw: str, k: int) -> str:
    s = str(raw).upper().replace("ELEM_", "")
    if s.startswith("FE_"):
        s = s[3:]
    if s in {"HEX", "HEXA", "BRICK"}:
        return "HEX20" if k >= 20 else "HEX8"
    if s in {"TET", "TETRA"}:
        return "TET10" if k >= 10 else "TET4"
    if s in {"WEDGE", "PENTA"}:
        return "QUADRATIC_WEDGE" if k >= 15 else "WEDGE"
    if s in {"PYRA", "PYRAMID"}:
        return "PYRA5"
    if s in {"QUAD", "QUAD4"}:
        return "QUAD8" if k >= 8 else "QUAD4"
    if s in {"TRI", "TRIANGLE"}:
        return "QUADRATIC_TRIANGLE" if k >= 6 else "TRI3"
    if s in {"LINE"}:
        return "LINE3" if k >= 3 else "LINE2"
    return s


def _vtk_cell_type(etype: str, k: int) -> int | None:
    return _VTK_MAP.get((_norm_label(etype, k), k))


def _six_to_nine(voigt6: np.ndarray) -> np.ndarray:
    """
    Input: (N,6) with order [xx, yy, zz, yz, xz, xy]
    Output: (N,9) row-major [xx, xy, xz, yx, yy, yz, zx, zy, zz]
    """
    v = np.asarray(voigt6, dtype=np.float32)
    out = np.empty((v.shape[0], 9), dtype=np.float32)
    xx, yy, zz, yz, xz, xy = v.T
    out[:, 0] = xx
    out[:, 1] = xy
    out[:, 2] = xz
    out[:, 3] = xy
    out[:, 4] = yy
    out[:, 5] = yz
    out[:, 6] = xz
    out[:, 7] = yz
    out[:, 8] = zz
    return out


def _set_point_data_vectors(ds: pv.DataSet, name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr, dtype=np.float32)
    ds.point_data[name] = a
    ds.set_active_vectors(name)


def _set_cell_data_vectors(ds: pv.DataSet, name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr, dtype=np.float32)
    ds.cell_data[name] = a
    ds.set_active_vectors(name)


def _set_point_data_tensors(ds: pv.DataSet, name: str, arr9: np.ndarray) -> None:
    a = np.asarray(arr9, dtype=np.float32)
    ds.point_data[name] = a
    ds.set_active_tensors(name)


def _set_cell_data_tensors(ds: pv.DataSet, name: str, arr9: np.ndarray) -> None:
    a = np.asarray(arr9, dtype=np.float32)
    ds.cell_data[name] = a
    ds.set_active_tensors(name)


def _set_point_data_scalars(ds: pv.DataSet, name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    ds.point_data[name] = a
    ds.set_active_scalars(name)


def _set_cell_data_scalars(ds: pv.DataSet, name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    ds.cell_data[name] = a
    ds.set_active_scalars(name)


# -------------------------- public bridge --------------------------


@dataclass
class PVBridge:
    """Build per-domain grids and attach results to PyVista datasets."""

    mesh: Mesh

    # ---------- grids ----------

    def domain_grid(self, domain: str) -> pv.UnstructuredGrid:
        m = self.mesh
        if domain not in m.parts:
            raise KeyError(f"unknown domain '{domain}'")
        rows = np.asarray(m.parts[domain], dtype=np.int64)
        if rows.size == 0:
            raise ValueError(f"domain '{domain}' has no elements")

        xyz = np.asarray(m.nodes.xyz, dtype=np.float64, order="C")
        conn = np.asarray(m.elements.conn, dtype=np.int64, order="C")
        nper = np.asarray(m.elements.nper, dtype=np.int64, order="C")
        etype = np.asarray(m.elements.etype, dtype=object)

        cells_list: List[np.ndarray] = []
        ctype_list: List[int] = []

        for e in rows:
            k = int(nper[e])
            if k <= 0:
                continue
            vtk_id = _vtk_cell_type(str(etype[e]), k)
            if vtk_id is None:
                continue
            ids = conn[e, :k]
            if ids.min() < 0:
                continue
            rec = np.concatenate(([k], ids.astype(np.int64)))
            cells_list.append(rec)
            ctype_list.append(vtk_id)

        if not ctype_list:
            raise ValueError(f"domain '{domain}' has no supported cells")

        cells = np.ascontiguousarray(np.concatenate(cells_list).astype(np.int64))
        ctypes = np.ascontiguousarray(np.array(ctype_list, dtype=np.uint8))
        return pv.UnstructuredGrid(cells, ctypes, xyz).clean(tolerance=0.0)

    def surface_mesh(self, surface: str) -> pv.PolyData:
        m = self.mesh
        if surface not in m.surfaces:
            raise KeyError(f"unknown surface '{surface}'")
        sa: SurfaceArray = m.surfaces[surface]
        faces_list: List[np.ndarray] = []
        for f in range(sa.faces.shape[0]):
            k = int(sa.nper[f])
            if k <= 0:
                continue
            ids = sa.faces[f, :k]
            faces_list.append(np.concatenate(([k], ids.astype(np.int64))))
        if not faces_list:
            raise ValueError(f"surface '{surface}' has no faces")
        faces = np.ascontiguousarray(np.concatenate(faces_list).astype(np.int64))
        pts = np.asarray(m.nodes.xyz, dtype=np.float64, order="C")
        return pv.PolyData(pts, faces).clean()

    # ---------- node data ----------

    def add_node_result(
        self,
        ds: pv.DataSet,
        view: NodeResultView,
        t: int,
        name: str | None = None,
        comp: int | slice | Iterable[int] | np.ndarray | str | None = ":",
    ) -> str:
        """
        Attaches nodal data to `ds.point_data` at time index `t`.
        Returns array name used.
        """
        arr = view[t, ":", comp]
        n, c = (arr.shape[0], 1) if arr.ndim == 1 else arr.shape
        out_name = name or view.meta.name

        if view.meta.dtype == FEDataType.VEC3F and c == 3:
            _set_point_data_vectors(ds, out_name, arr.reshape(n, 3))
        elif view.meta.dtype in (FEDataType.MAT3FS, FEDataType.MAT3F):
            if c == 1:
                _set_point_data_scalars(ds, out_name, arr.reshape(n))
            elif c == 6 and view.meta.dtype == FEDataType.MAT3FS:
                _set_point_data_tensors(ds, out_name, _six_to_nine(arr.reshape(n, 6)))
            elif c == 9:
                _set_point_data_tensors(ds, out_name, arr.reshape(n, 9))
            else:
                # fallback: first component as scalar
                _set_point_data_scalars(ds, out_name, arr.reshape(n, c)[:, 0])
        else:
            _set_point_data_scalars(ds, out_name, arr.reshape(n))
        return out_name

    # ---------- per-item data (elements/faces) ----------

    def add_elem_item_result(
        self,
        ds: pv.UnstructuredGrid,
        view: ItemResultView,
        domain: str,
        t: int,
        name: str | None = None,
        comp: int | slice | Iterable[int] | np.ndarray | str | None = ":",
    ) -> str:
        """
        Attaches element FMT_ITEM data for `domain` to `ds.cell_data`.
        `ds` must be the grid built for the same domain.
        """
        local = view.region(domain)
        arr = local[t, ":", comp]  # (R,C) or (R,)
        r, c = (arr.shape[0], 1) if arr.ndim == 1 else arr.shape
        if ds.n_cells != r:
            raise ValueError(
                f"grid cells {ds.n_cells} != result rows {r} for domain '{domain}'"
            )
        out_name = name or view.meta.name

        if view.meta.dtype == FEDataType.VEC3F and c == 3:
            _set_cell_data_vectors(ds, out_name, arr.reshape(r, 3))
        elif view.meta.dtype in (FEDataType.MAT3FS, FEDataType.MAT3F):
            if c == 1:
                _set_cell_data_scalars(ds, out_name, arr.reshape(r))
            elif c == 6 and view.meta.dtype == FEDataType.MAT3FS:
                _set_cell_data_tensors(ds, out_name, _six_to_nine(arr.reshape(r, 6)))
            elif c == 9:
                _set_cell_data_tensors(ds, out_name, arr.reshape(r, 9))
            else:
                _set_cell_data_scalars(ds, out_name, arr.reshape(r, c)[:, 0])
        else:
            _set_cell_data_scalars(ds, out_name, arr.reshape(r))
        return out_name

    def add_face_item_result(
        self,
        ds: pv.PolyData | pv.UnstructuredGrid,
        view: ItemResultView,
        surface: str,
        t: int,
        name: str | None = None,
        comp: int | slice | Iterable[int] | np.ndarray | str | None = ":",
    ) -> str:
        """
        Attaches surface FMT_ITEM data for `surface` to `cell_data` of `ds`.
        """
        local = view.region(surface)
        arr = local[t, ":", comp]
        r, c = (arr.shape[0], 1) if arr.ndim == 1 else arr.shape
        if ds.n_cells != r:
            raise ValueError(
                f"mesh faces {ds.n_cells} != result rows {r} for surface '{surface}'"
            )
        out_name = name or view.meta.name

        if view.meta.dtype == FEDataType.VEC3F and c == 3:
            _set_cell_data_vectors(ds, out_name, arr.reshape(r, 3))
        elif view.meta.dtype in (FEDataType.MAT3FS, FEDataType.MAT3F):
            if c == 1:
                _set_cell_data_scalars(ds, out_name, arr.reshape(r))
            elif c == 6 and view.meta.dtype == FEDataType.MAT3FS:
                _set_cell_data_tensors(ds, out_name, _six_to_nine(arr.reshape(r, 6)))
            elif c == 9:
                _set_cell_data_tensors(ds, out_name, arr.reshape(r, 9))
            else:
                _set_cell_data_scalars(ds, out_name, arr.reshape(r, c)[:, 0])
        else:
            _set_cell_data_scalars(ds, out_name, arr.reshape(r))
        return out_name

    # ---------- FMT_MULT reduction to cells ----------

    def add_elem_mult_reduced(
        self,
        ds: pv.UnstructuredGrid,
        view: MultResultView,
        domain: str,
        t: int,
        reducer: str = "mean",
        name: str | None = None,
        comp: int | slice | Iterable[int] | np.ndarray | str | None = ":",
    ) -> str:
        """
        Reduces FMT_MULT (per element-node) to one value per element and
        attaches to `cell_data`.
        Reducers: 'mean', 'max', 'min', 'first'.
        """
        local = view.region(domain)
        blk = local[t, ":", ":", comp]  # (R, Kmax, C) or (R, Kmax)
        if blk.ndim == 2:
            blk = blk[:, :, None]
        R, Kmax, C = blk.shape
        if ds.n_cells != R:
            raise ValueError(
                f"grid cells {ds.n_cells} != result rows {R} for domain '{domain}'"
            )

        elem_rows = np.asarray(self.mesh.parts[domain], dtype=np.int64)
        nper = np.asarray(self.mesh.elements.nper[elem_rows], dtype=np.int64)

        red_fn: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
            "mean": lambda a, k: np.nanmean(a[:k, :], axis=0),
            "max": lambda a, k: np.nanmax(a[:k, :], axis=0),
            "min": lambda a, k: np.nanmin(a[:k, :], axis=0),
            "first": lambda a, k: a[0:1, :].reshape(-1)
            if k > 0
            else np.full((a.shape[1],), np.nan, np.float32),
        }

        if reducer not in red_fn:
            raise ValueError("reducer must be one of {'mean','max','min','first'}")

        out = np.empty((R, C), dtype=np.float32)
        for r in range(R):
            k = int(nper[r])
            out[r, :] = red_fn[reducer](blk[r, :, :], k)

        out_name = name or f"{view.meta.name}_{reducer}"
        # vector/tensor handling mirrors ITEM
        if view.meta.dtype == FEDataType.VEC3F and C == 3:
            _set_cell_data_vectors(ds, out_name, out.reshape(R, 3))
        elif view.meta.dtype in (FEDataType.MAT3FS, FEDataType.MAT3F):
            if C == 1:
                _set_cell_data_scalars(ds, out_name, out.reshape(R))
            elif C == 6 and view.meta.dtype == FEDataType.MAT3FS:
                _set_cell_data_tensors(ds, out_name, _six_to_nine(out.reshape(R, 6)))
            elif C == 9:
                _set_cell_data_tensors(ds, out_name, out.reshape(R, 9))
            else:
                _set_cell_data_scalars(ds, out_name, out[:, 0])
        else:
            _set_cell_data_scalars(ds, out_name, out.reshape(R))
        return out_name

    # ---------- region vectors ----------

    def region_series(
        self,
        view: RegionResultView,
        region: str,
        comp: int | slice | Iterable[int] | np.ndarray | str | None = ":",
    ) -> np.ndarray:
        """
        Returns a (T, C_sel) array of region (domain/surface) values over time.
        Useful for custom plotting (not attached to a mesh).
        """
        return (
            view.time(":").eval(region=region)[..., _normalize_comp(comp)]
            if comp != ":"
            else view.time(":").eval(region=region)
        )


# tiny helper to normalize component argument when slicing numpy arrays directly
def _normalize_comp(
    comp: int | slice | Iterable[int] | np.ndarray | str | None,
) -> int | slice | Iterable[int] | np.ndarray | None:
    return None if comp == ":" else comp
