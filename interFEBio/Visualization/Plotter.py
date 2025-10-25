from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pyvista as pv
from interFEBio.Mesh.Mesh import Mesh, SurfaceArray
from interFEBio.XPLT.Enums import FEDataType
from interFEBio.XPLT.XPLT import (
    ItemResultView,
    MultResultView,
    NodeResultView,
    RegionResultView,
)

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
    # (N,6)[xx,yy,zz,yz,xz,xy] -> (N,9) row-major 3x3
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


def _attach_point(ds: pv.DataSet, name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr, dtype=np.float32)
    ds.point_data[name] = a


def _attach_cell(ds: pv.DataSet, name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr, dtype=np.float32)
    ds.cell_data[name] = a


# -------------------------- public bridge --------------------------


@dataclass
class PVBridge:
    """Build per-domain grids and attach pre-sliced results to PyVista."""

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
            cells_list.append(np.concatenate(([k], ids.astype(np.int64))))
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

    # ---------- node data (array provided) ----------

    def add_node_result_array(
        self,
        ds: pv.DataSet,
        *,
        view: NodeResultView,
        data: np.ndarray,
        name: str | None = None,
        set_active: bool = True,
    ) -> str:
        """
        Attach a pre-sliced nodal array to point_data.
        data shape: (N,) or (N, C). N must equal ds.n_points.
        """
        a = np.asarray(data, dtype=np.float32)
        if a.ndim == 1:
            if a.shape[0] != ds.n_points:
                raise ValueError(f"points {ds.n_points} != data {a.shape[0]}")
            _attach_point(ds, name or view.meta.name, a)
            if set_active:
                ds.set_active_scalars(name or view.meta.name)
            return name or view.meta.name

        if a.shape[0] != ds.n_points:
            raise ValueError(f"points {ds.n_points} != data {a.shape[0]}")
        c = a.shape[1]
        nm = name or view.meta.name

        if view.meta.dtype == FEDataType.VEC3F and c == 3:
            _attach_point(ds, nm, a)
            if set_active:
                ds.set_active_vectors(nm)
        elif view.meta.dtype in (FEDataType.MAT3FS, FEDataType.MAT3F):
            if c == 6 and view.meta.dtype == FEDataType.MAT3FS:
                _attach_point(ds, nm, _six_to_nine(a))
                if set_active:
                    ds.set_active_tensors(nm)
            elif c == 9:
                _attach_point(ds, nm, a)
                if set_active:
                    ds.set_active_tensors(nm)
            else:
                _attach_point(ds, nm, a[:, 0])
                if set_active:
                    ds.set_active_scalars(nm)
        else:
            _attach_point(ds, nm, a[:, 0] if c > 1 else a)
            if set_active:
                ds.set_active_scalars(nm)
        return nm

    # ---------- element FMT_ITEM (array provided) ----------

    def add_elem_item_array(
        self,
        ds: pv.UnstructuredGrid,
        *,
        view: ItemResultView,
        domain: str,
        data: np.ndarray,
        name: str | None = None,
        set_active: bool = True,
    ) -> str:
        """
        Attach a pre-sliced per-element array to cell_data.
        data shape: (R,) or (R, C). R must equal ds.n_cells.
        """
        a = np.asarray(data, dtype=np.float32)
        if a.ndim == 1:
            if a.shape[0] != ds.n_cells:
                raise ValueError(f"cells {ds.n_cells} != data {a.shape[0]}")
            _attach_cell(ds, name or view.meta.name, a)
            if set_active:
                ds.set_active_scalars(name or view.meta.name)
            return name or view.meta.name

        if a.shape[0] != ds.n_cells:
            raise ValueError(f"cells {ds.n_cells} != data {a.shape[0]}")
        c = a.shape[1]
        nm = name or view.meta.name

        if view.meta.dtype == FEDataType.VEC3F and c == 3:
            _attach_cell(ds, nm, a)
            if set_active:
                ds.set_active_vectors(nm)
        elif view.meta.dtype in (FEDataType.MAT3FS, FEDataType.MAT3F):
            if c == 6 and view.meta.dtype == FEDataType.MAT3FS:
                _attach_cell(ds, nm, _six_to_nine(a))
                if set_active:
                    ds.set_active_tensors(nm)
            elif c == 9:
                _attach_cell(ds, nm, a)
                if set_active:
                    ds.set_active_tensors(nm)
            else:
                _attach_cell(ds, nm, a[:, 0])
                if set_active:
                    ds.set_active_scalars(nm)
        else:
            _attach_cell(ds, nm, a[:, 0] if c > 1 else a)
            if set_active:
                ds.set_active_scalars(nm)
        return nm

    # ---------- surface FMT_ITEM (array provided) ----------

    def add_face_item_array(
        self,
        ds: pv.PolyData | pv.UnstructuredGrid,
        *,
        view: ItemResultView,
        surface: str,
        data: np.ndarray,
        name: str | None = None,
        set_active: bool = True,
    ) -> str:
        """
        Attach a pre-sliced per-face array to cell_data.
        data shape: (R,) or (R, C). R must equal ds.n_cells.
        """
        a = np.asarray(data, dtype=np.float32)
        if a.ndim == 1:
            if a.shape[0] != ds.n_cells:
                raise ValueError(f"cells {ds.n_cells} != data {a.shape[0]}")
            _attach_cell(ds, name or view.meta.name, a)
            if set_active:
                ds.set_active_scalars(name or view.meta.name)
            return name or view.meta.name

        if a.shape[0] != ds.n_cells:
            raise ValueError(f"cells {ds.n_cells} != data {a.shape[0]}")
        c = a.shape[1]
        nm = name or view.meta.name

        if view.meta.dtype == FEDataType.VEC3F and c == 3:
            _attach_cell(ds, nm, a)
            if set_active:
                ds.set_active_vectors(nm)
        elif view.meta.dtype in (FEDataType.MAT3FS, FEDataType.MAT3F):
            if c == 6 and view.meta.dtype == FEDataType.MAT3FS:
                _attach_cell(ds, nm, _six_to_nine(a))
                if set_active:
                    ds.set_active_tensors(nm)
            elif c == 9:
                _attach_cell(ds, nm, a)
                if set_active:
                    ds.set_active_tensors(nm)
            else:
                _attach_cell(ds, nm, a[:, 0])
                if set_active:
                    ds.set_active_scalars(nm)
        else:
            _attach_cell(ds, nm, a[:, 0] if c > 1 else a)
            if set_active:
                ds.set_active_scalars(nm)
        return nm

    # ---------- FMT_MULT reduction (array provided) ----------

    def add_elem_mult_reduced_array(
        self,
        ds: pv.UnstructuredGrid,
        *,
        view: MultResultView,
        domain: str,
        data: np.ndarray,  # shape (R, Kmax) or (R, Kmax, C)
        reducer: str = "mean",
        name: str | None = None,
        set_active: bool = True,
    ) -> str:
        """
        Reduce a pre-packed per-element-node array into one value per element,
        then attach to cell_data.
        """
        blk = np.asarray(data, dtype=np.float32)
        if blk.ndim == 2:
            blk = blk[:, :, None]
        R, Kmax, C = blk.shape
        if ds.n_cells != R:
            raise ValueError(f"cells {ds.n_cells} != data rows {R}")

        elem_rows = np.asarray(self.mesh.parts[domain], dtype=np.int64)
        nper = np.asarray(self.mesh.elements.nper[elem_rows], dtype=np.int64)

        if reducer == "mean":
            out = np.vstack(
                [
                    np.nanmean(blk[r, : nper[r], :], axis=0)
                    if nper[r] > 0
                    else np.full((C,), np.nan, np.float32)
                    for r in range(R)
                ]
            )
        elif reducer == "max":
            out = np.vstack(
                [
                    np.nanmax(blk[r, : nper[r], :], axis=0)
                    if nper[r] > 0
                    else np.full((C,), np.nan, np.float32)
                    for r in range(R)
                ]
            )
        elif reducer == "min":
            out = np.vstack(
                [
                    np.nanmin(blk[r, : nper[r], :], axis=0)
                    if nper[r] > 0
                    else np.full((C,), np.nan, np.float32)
                    for r in range(R)
                ]
            )
        elif reducer == "first":
            out = np.vstack(
                [
                    blk[r, 0, :] if nper[r] > 0 else np.full((C,), np.nan, np.float32)
                    for r in range(R)
                ]
            )
        else:
            raise ValueError("reducer must be one of {'mean','max','min','first'}")

        nm = name or f"{view.meta.name}_{reducer}"
        if C == 1:
            _attach_cell(ds, nm, out.reshape(R))
            if set_active:
                ds.set_active_scalars(nm)
        elif view.meta.dtype == FEDataType.VEC3F and C == 3:
            _attach_cell(ds, nm, out.reshape(R, 3))
            if set_active:
                ds.set_active_vectors(nm)
        elif view.meta.dtype in (FEDataType.MAT3FS, FEDataType.MAT3F):
            if C == 6 and view.meta.dtype == FEDataType.MAT3FS:
                _attach_cell(ds, nm, _six_to_nine(out.reshape(R, 6)))
                if set_active:
                    ds.set_active_tensors(nm)
            elif C == 9:
                _attach_cell(ds, nm, out.reshape(R, 9))
                if set_active:
                    ds.set_active_tensors(nm)
            else:
                _attach_cell(ds, nm, out[:, 0])
                if set_active:
                    ds.set_active_scalars(nm)
        else:
            _attach_cell(ds, nm, out[:, 0])
            if set_active:
                ds.set_active_scalars(nm)
        return nm

    # ---------- region vectors (array provided) ----------

    def region_series_array(self, *, data: np.ndarray) -> np.ndarray:
        """
        Pass-through helper for pre-sliced region vectors over time.
        Ensures float32 and 2D shape (T, Csel).
        """
        a = np.asarray(data, dtype=np.float32)
        return a.reshape(a.shape[0], -1)
