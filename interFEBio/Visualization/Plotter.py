from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import pyvista as pv
from interFEBio.Mesh.Mesh import Mesh

# ---------- Helper functions ----------


def _norm_label(raw: str, k: int) -> str:
    # Remove the 'ELEM_' prefix and handle the rest
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


def _is_2d_or_1d(vtk_id: int) -> bool:
    return vtk_id in (
        pv.CellType.QUAD,
        pv.CellType.QUADRATIC_QUAD,
        pv.CellType.TRIANGLE,
        pv.CellType.QUADRATIC_TRIANGLE,
        pv.CellType.LINE,
        pv.CellType.QUADRATIC_EDGE,
    )


# ---------- Cache and Plotter Class ----------


@dataclass
class _Cache:
    point_grid: pv.DataSet | None = None
    surface_poly_by_name: Dict[str, pv.PolyData] = field(default_factory=dict)


class PVPlotter:
    def __init__(self) -> None:
        self._mesh: Mesh | None = None
        self._xplt: object | None = None
        self._cache = _Cache()
        self._elem_domain_of: List[str] | None = None
        self._eid2row: Dict[int, int] | None = None

    def attach_mesh(self, mesh: Mesh) -> None:
        """Attach a mesh to the plotter."""
        self._mesh = mesh
        self._xplt = None
        self._cache = _Cache()
        self._eid2row = None
        self._build_elem_domain_map()

    def attach_xplt(self, xp: object) -> None:
        """Attach an XPLT object to the plotter."""
        self._xplt = xp
        self._mesh = getattr(xp, "mesh")
        self._cache = _Cache()
        self._eid2row = None
        self._build_elem_domain_map()

    def _elem_id_array(self) -> np.ndarray | None:
        """Get element IDs from the mesh."""
        if self._mesh is None:
            return None
        for attr in ("ids", "elem_ids", "id"):
            if hasattr(self._mesh.elements, attr):
                a = np.asarray(getattr(self._mesh.elements, attr))
                if a.size:
                    return a.astype(np.int64, copy=False)
        return None

    def _ensure_eid_map(self) -> None:
        """Ensure the element ID map exists."""
        if self._eid2row is not None:
            return
        eid = self._elem_id_array()
        self._eid2row = (
            {int(eid[i]): i for i in range(eid.shape[0])} if eid is not None else {}
        )

    def _map_elem_refs_to_rows(self, refs: np.ndarray) -> np.ndarray:
        """Map element references to rows in the element array."""
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

    def _build_elem_domain_map(self) -> None:
        """Map elements to their respective domains."""
        if self._mesh is None:
            self._elem_domain_of = None
            return
        E = self._mesh.elements.conn.shape[0]
        dom_of = np.full(E, "", dtype=object)
        for name, idx in self._mesh.parts.items():
            ids = self._map_elem_refs_to_rows(np.asarray(idx, dtype=np.int64))
            dom_of[ids] = name
        self._elem_domain_of = dom_of.tolist()

    def _build_point_grid(self, domain: str) -> pv.DataSet:
        """Build a point grid from the mesh for the specified domain."""
        if self._mesh is None:
            raise RuntimeError("Mesh not attached")

        # Extract the specific domain's elements and nodes
        m = self._mesh
        xyz = np.ascontiguousarray(np.asarray(m.nodes.xyz, dtype=np.float64))
        conn = np.asarray(m.elements.conn, dtype=np.int64)
        nper = np.asarray(m.elements.nper, dtype=np.int64)
        etype = np.asarray(m.elements.etype, dtype=object)

        # Create the mapping for the cell types (to map elements to PyVista cell types)
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

        # We now need to filter the domain's elements
        domain_elements = m.parts.get(domain, [])

        # Ensure that domain_elements is not empty
        if len(domain_elements) == 0:
            raise ValueError(f"Domain '{domain}' not found or contains no elements.")

        pieces3d: List[np.ndarray] = []
        types3d: List[int] = []
        pieces2d1d: List[np.ndarray] = []
        types2d1d: List[int] = []

        # Process the elements for the given domain
        for e in domain_elements:
            k = int(nper[e])
            if k <= 0:
                continue
            lbl = _norm_label(etype[e], k)
            vtk_id = map_vtk.get(lbl)
            if vtk_id is None:
                print(f"Skipping element {e} with unsupported type: {lbl}")
                continue
            ids = conn[e, :k]
            if ids.min() < 0 or ids.max() >= xyz.shape[0]:
                continue
            rec = np.concatenate(([k], ids.astype(np.int64)))

            if vtk_id in {
                pv.CellType.TETRA,
                pv.CellType.HEXAHEDRON,
                pv.CellType.WEDGE,
                pv.CellType.PYRAMID,
            }:
                pieces3d.append(rec)
                types3d.append(vtk_id)
            elif vtk_id in {pv.CellType.QUAD, pv.CellType.TRIANGLE, pv.CellType.LINE}:
                pieces2d1d.append(rec)
                types2d1d.append(vtk_id)

        # Create volume grid (3D elements)
        vol_grid = None
        if types3d:
            cells3d = np.ascontiguousarray(np.concatenate(pieces3d).astype(np.int64))
            ctype3d = np.ascontiguousarray(np.array(types3d, dtype=np.uint8))
            vol_grid = pv.UnstructuredGrid(cells3d, ctype3d, xyz).clean(tolerance=0.0)

        # Create surface geometry (2D or 1D elements)
        geom_poly = None
        if types2d1d:
            cells2d = np.ascontiguousarray(np.concatenate(pieces2d1d).astype(np.int64))
            ctypes2d = np.ascontiguousarray(np.array(types2d1d, dtype=np.uint8))
            geom_poly = (
                pv.UnstructuredGrid(cells2d, ctypes2d, xyz).extract_geometry().clean()
            )

        return vol_grid, geom_poly

    def get_domain_data(self, domain: str) -> Tuple[pv.DataSet, pv.PolyData]:
        """Get domain-specific data (volume grid + surface)."""
        vol_grid, geom_poly = self._build_point_grid()

        # Extract the surface for the given domain
        if domain != "__shells__":
            return vol_grid, geom_poly

        return vol_grid, self._cache.surface_poly_by_name.get("__shells__")

    def get_all_domains(self) -> List[str]:
        """Get all available domain names."""
        return list(self._mesh.parts.keys())

    def get_domains_for_plot(self) -> List[Tuple[pv.DataSet, pv.PolyData]]:
        """Get all domains' data ready for plotting."""
        domains = self.get_all_domains()
        domain_data = []
        for domain in domains:
            domain_data.append(self.get_domain_data(domain))
        return domain_data

    def get_single_domain_data(self, domain: str) -> pv.DataSet:
        """Get the appropriate domain data (volume grid or surface) for plotting."""
        # Retrieve the domain data (volume grid or surface) for the given domain
        vol_grid, geom_poly = self._build_point_grid(domain)

        # Return the appropriate mesh data based on the domain's element types
        if vol_grid:
            print(f"Domain '{domain}' contains 3D elements, returning vol_grid.")
            return vol_grid
        if geom_poly:
            print(f"Domain '{domain}' contains surface elements, returning geom_poly.")
            return geom_poly

        print(f"Domain '{domain}' has no valid mesh data.")
        return None  # If neither vol_grid nor geom_poly is available
