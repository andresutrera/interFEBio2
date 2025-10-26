from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Sequence,
)

import numpy as np

if TYPE_CHECKING:
    from interFEBio.XPLT import xplt

# ----------------------------------------------------------------------
# Low-level tables with slice-first APIs
# ----------------------------------------------------------------------


@dataclass
class NodeArray:
    """Dense node table with slice-friendly access.

    Attributes
    ----------
    xyz
        Array of shape (N, 3). Stored as float64.
    """

    xyz: np.ndarray

    def __post_init__(self) -> None:
        a = np.asanyarray(self.xyz, dtype=float)
        if a.ndim != 2:
            raise ValueError("nodes must be 2D")
        if a.shape[1] == 2:
            a = np.hstack((a, np.zeros((a.shape[0], 1), dtype=a.dtype)))
        elif a.shape[1] > 3:
            a = a[:, :3]
        self.xyz = a

    def __len__(self) -> int:
        """Number of nodes."""
        return int(self.xyz.shape[0])

    def __getitem__(self, idx: slice | int | np.ndarray | list[int]) -> NodeArray:
        """Return a sliced view."""
        return NodeArray(self.xyz[idx])

    def take(self, ids: np.ndarray | list[int]) -> NodeArray:
        """Return a new node table with selected node ids."""
        return NodeArray(self.xyz[np.asarray(ids, dtype=np.int64)])


@dataclass
class ElementArray:
    """Mixed-element connectivity with padding for slice simplicity.

    Elements are stored in a 2D integer array with padding using ``-1``.
    The actual number of nodes per element is given by ``nper``. This
    keeps slicing trivial while still supporting variable arity.

    Attributes
    ----------
    conn
        Array of shape (E, Kmax) with 0-based node ids. Unused slots are ``-1``.
    nper
        Array of shape (E,) with the valid count per row.
    etype
        Array of shape (E,) with canonical labels such as ``'hex8'``, ``'tet4'``.
    """

    conn: np.ndarray
    nper: np.ndarray
    etype: np.ndarray

    def __post_init__(self) -> None:
        c = np.asanyarray(self.conn, dtype=np.int64)
        if c.ndim != 2:
            raise ValueError("conn must be (E, Kmax)")
        self.conn = c
        self.nper = np.asanyarray(self.nper, dtype=np.int64).reshape(-1)
        self.etype = np.asanyarray(self.etype)
        if (
            self.conn.shape[0] != self.nper.shape[0]
            or self.conn.shape[0] != self.etype.shape[0]
        ):
            raise ValueError("conn, nper, etype length mismatch")

    def __len__(self) -> int:
        """Number of elements."""
        return int(self.conn.shape[0])

    def __getitem__(self, idx: slice | int | np.ndarray | list[int]) -> ElementArray:
        """Return a sliced view."""
        return ElementArray(self.conn[idx], self.nper[idx], self.etype[idx])

    def nodes_of(self, ei: int) -> np.ndarray:
        """Return 1D array of node ids for a single element."""
        k = self.nper[ei]
        return self.conn[ei, :k]

    def unique_nodes(self) -> np.ndarray:
        """Return sorted unique node ids across the selection."""
        if len(self) == 0:
            return np.empty((0,), dtype=np.int64)
        valid = self._mask_valid()
        return np.unique(self.conn[valid])

    def as_ragged(self) -> list[np.ndarray]:
        """Return Python list of 1D arrays per element."""
        return [self.nodes_of(i) for i in range(len(self))]

    def _mask_valid(self) -> np.ndarray:
        """Boolean mask same shape as conn with True on valid entries."""
        counts = np.repeat(self.nper[:, None], self.conn.shape[1], axis=1)
        idx = np.repeat(np.arange(self.conn.shape[1])[None, :], len(self), axis=0)
        return idx < counts


@dataclass
class SurfaceArray:
    """Facet connectivity with padding for slice simplicity.

    Attributes
    ----------
    faces
        Array of shape (F, Kmax) with 0-based node ids. Unused slots are ``-1``.
    nper
        Array of shape (F,) with valid count per facet.
    """

    faces: np.ndarray
    nper: np.ndarray

    def __post_init__(self) -> None:
        f = np.asanyarray(self.faces, dtype=np.int64)
        if f.ndim != 2:
            raise ValueError("faces must be (F, Kmax)")
        self.faces = f
        self.nper = np.asanyarray(self.nper, dtype=np.int64).reshape(-1)
        if self.faces.shape[0] != self.nper.shape[0]:
            raise ValueError("faces and nper length mismatch")

    def __len__(self) -> int:
        """Number of facets."""
        return int(self.faces.shape[0])

    def __getitem__(self, idx: slice | int | np.ndarray | list[int]) -> SurfaceArray:
        """Return a sliced view."""
        return SurfaceArray(self.faces[idx], self.nper[idx])

    def nodes_of(self, fi: int) -> np.ndarray:
        """Return node ids for a single facet."""
        k = self.nper[fi]
        return self.faces[fi, :k]

    def unique_nodes(self) -> np.ndarray:
        """Return sorted unique node ids across the selection."""
        if len(self) == 0:
            return np.empty((0,), dtype=np.int64)
        valid = self._mask_valid()
        return np.unique(self.faces[valid])

    def _mask_valid(self) -> np.ndarray:
        counts = np.repeat(self.nper[:, None], self.faces.shape[1], axis=1)
        idx = np.repeat(np.arange(self.faces.shape[1])[None, :], len(self), axis=0)
        return idx < counts


# ----------------------------------------------------------------------
# Mesh container with isolated subsystems and name-first access
# ----------------------------------------------------------------------


class Mesh:
    """Unified mesh container with isolated node, element, and surface tables.

    Keeps each concern in its own class for maintainability. Still allows
    NumPy slicing on nodes and elements while avoiding CSR complexity.

    Attributes
    ----------
    nodes
        Node table.
    elements
        Element table for the full mesh.
    parts
        Mapping ``name -> element indices`` selecting rows of ``elements``.
    surfaces
        Mapping ``name -> SurfaceArray``.
    nodesets
        Mapping ``name -> node ids``.
    """

    nodes: NodeArray
    elements: ElementArray
    parts: dict[str, np.ndarray]
    surfaces: dict[str, SurfaceArray]
    nodesets: dict[str, np.ndarray]

    # -------------- construction --------------

    def __init__(
        self,
        nodes: NodeArray,
        elements: ElementArray,
        parts: dict[str, np.ndarray] | None = None,
        surfaces: dict[str, SurfaceArray] | None = None,
        nodesets: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Initialize the mesh."""
        self.nodes = nodes
        self.elements = elements
        self.parts = (
            {}
            if parts is None
            else {k: np.asarray(v, dtype=np.int64) for k, v in parts.items()}
        )
        self.surfaces = {} if surfaces is None else dict(surfaces)
        self.nodesets = (
            {}
            if nodesets is None
            else {k: np.asarray(v, dtype=np.int64) for k, v in nodesets.items()}
        )

    # -------------- simple queries --------------

    @property
    def nelems(self) -> int:
        """Number of elements in the mesh."""
        return len(self.elements)

    @property
    def nnodes(self) -> int:
        """Number of nodes in the mesh."""
        return len(self.nodes)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box."""
        return self.nodes.xyz.min(0), self.nodes.xyz.max(0)

    # -------------- name-first views --------------

    def getDomain(self, name: str) -> ElementArray:
        """Return a slice of ``elements`` for a named part.

        Args
        ----
        name
            Part name.

        Returns
        -------
        ElementArray
            Slice view for the requested domain.
        """
        return self.elements[self.parts[name]]

    def getSurface(self, name: str) -> SurfaceArray:
        """Return a surface by name.

        Args
        ----
        name
            Surface name.

        Returns
        -------
        SurfaceArray
            Facet table.
        """
        return self.surfaces[name]

    def getNodeset(self, name: str) -> NodeArray:
        """Return a nodeset by name as a node slice.

        Args
        ----
        name
            Nodeset name.

        Returns
        -------
        NodeArray
            Node slice view.
        """
        return self.nodes.take(self.nodesets[name])

    # -------------- FEBio XML writers --------------

    def to_feb_nodes_xml(self, object_name: str = "Object1") -> ET.Element:
        """Build a FEBio ``<Nodes>`` element.

        Args
        ----
        object_name
            FEBio object name.

        Returns
        -------
        Element
            Root ``<Nodes>`` element.
        """
        root = ET.Element("Nodes", name=object_name)
        for i, xyz in enumerate(self.nodes.xyz, start=1):
            node_el = ET.Element("node", id=str(i))
            node_el.text = ",".join(map(str, xyz))
            root.append(node_el)
        return root

    def to_feb_elements_xml(self) -> list[ET.Element]:
        """Build one FEBio ``<Elements>`` element per part.

        Returns
        -------
        list
            list of ``<Elements>`` elements.
        """
        out: list[ET.Element] = []
        for pname, eidx in self.parts.items():
            part = self.elements[eidx]
            # Decide type label: if mixed, use first
            etype = str(np.unique(part.etype)[0]) if len(part) else "unknown"
            el = ET.Element("Elements", type=etype, name=pname)
            for j in range(len(part)):
                nodes = (part.nodes_of(j) + 1).tolist()
                e_el = ET.Element("elem", id=str(j + 1))
                e_el.text = ",".join(map(str, nodes))
                el.append(e_el)
            out.append(el)
        return out

    def to_feb_surfaces_xml(self) -> list[ET.Element]:
        """Build one FEBio ``<Surface>`` per named surface.

        Returns
        -------
        list
            list of ``<Surface>`` elements.
        """
        out: list[ET.Element] = []
        for sname, surf in self.surfaces.items():
            el = ET.Element("Surface", name=sname)
            for i in range(len(surf)):
                nn = surf.nper[i]
                etype = {2: "line2", 3: "tri3", 4: "quad4"}.get(int(nn), "facet")
                f_el = ET.Element(etype, id=str(i + 1))
                f_el.text = ",".join(map(str, (surf.nodes_of(i) + 1).tolist()))
                el.append(f_el)
            out.append(el)
        return out

    def to_feb_nodesets_xml(self) -> list[ET.Element]:
        """Build FEBio ``<NodeSet>`` elements.

        Returns
        -------
        list
            list of ``<NodeSet>`` elements.
        """
        out: list[ET.Element] = []
        for nname, nids in self.nodesets.items():
            el = ET.Element("NodeSet", name=nname)
            el.text = ",".join(map(str, (nids + 1).tolist()))
            out.append(el)
        return out

    # -------------- constructors from sources --------------

    @classmethod
    def from_gmsh_msh(
        cls,
        path: str,
        include: Sequence[str] | None = None,
        scale: Sequence[float] | np.ndarray = (1.0, 1.0, 1.0),
        quiet: bool = False,
    ) -> Mesh:
        """Create from a Gmsh 2.0/2.2 ASCII ``.msh`` file.

        Args
        ----
        path
            Path to file.
        include
            Physical names to include as parts. If None, include all volumes.
        scale
            Per-axis scale for node coordinates.
        quiet
            Suppress warnings.

        Returns
        -------
        Mesh
            New instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f]

        def _section(tag: str) -> list[str]:
            a = lines.index(f"${tag}") + 1
            n = int(lines[a])
            return lines[a + 1 : a + 1 + n]

        fmt = lines[lines.index("$MeshFormat") + 1].split()[0]
        if fmt not in ("2.0", "2.2"):
            sys.exit(f"Error: Gmsh version {fmt} not supported")

        # Physical names
        dim_map = {0: "node", 1: "line", 2: "surface", 3: "volume"}
        phys_raw = _section("PhysicalNames")
        phys: dict[int, tuple[str, str]] = {}
        for r in phys_raw:
            d, pid, name = int(r.split()[0]), int(r.split()[1]), r.split('"')[1]
            phys[pid] = (name, dim_map[d])

        # Nodes
        sx, sy, sz = map(float, scale)
        node_rows = _section("Nodes")
        ids, coords = [], []
        for r in node_rows:
            t = r.split()
            ids.append(int(t[0]))
            coords.append((float(t[1]) * sx, float(t[2]) * sy, float(t[3]) * sz))
        order = np.argsort(np.asarray(ids, dtype=np.int64))
        old2new = {int(ids[i]): int(j) for j, i in enumerate(order)}
        nodes = NodeArray(np.asarray(coords, dtype=float)[order])

        # Elements
        _GMSH_TO_ELEM = {
            1: "line2",
            2: "tri3",
            3: "quad4",
            4: "tet4",
            5: "hex8",
            6: "penta6",
            8: "line3",
            9: "tri6",
            10: "quad8",
            11: "tet10",
            12: "hex20",
            16: "penta15",
        }
        elem_rows = _section("Elements")
        conn_list: list[np.ndarray] = []
        etypes: list[str] = []
        part_map: dict[str, list[int]] = {}
        surf_map: dict[str, list[np.ndarray]] = {}
        nset_map: dict[str, set[int]] = {}

        include_set = set(include) if include is not None else None

        for r in elem_rows:
            t = r.split()
            etype_num = int(t[1])
            ntags = int(t[2])
            tags = list(map(int, t[3 : 3 + ntags]))
            phys_id = tags[0] if ntags else None
            raw_nodes = list(map(int, t[3 + ntags :]))

            etype = _GMSH_TO_ELEM.get(etype_num, "")
            if not etype:
                if not quiet:
                    print(f"Warning: skip unsupported gmsh type {etype_num}")
                continue
            if phys_id is None or phys_id not in phys:
                continue

            pname, ptype = phys[phys_id]
            new_nodes = np.fromiter((old2new[n] for n in raw_nodes), dtype=np.int64)
            if ptype == "volume":
                if include_set is None or pname in include_set:
                    conn_list.append(new_nodes)
                    etypes.append(etype)
                    part_map.setdefault(pname, []).append(len(etypes) - 1)
            elif ptype in ("surface", "line"):
                surf_map.setdefault(pname, []).append(new_nodes)
                nset_map.setdefault(pname, set()).update(map(int, new_nodes))

        # Pad connectivity
        kmax = max((len(c) for c in conn_list), default=0)
        conn = -np.ones((len(conn_list), kmax), dtype=np.int64)
        nper = np.zeros((len(conn_list),), dtype=np.int64)
        for i, c in enumerate(conn_list):
            conn[i, : c.size] = c
            nper[i] = c.size
        elements = ElementArray(
            conn=conn, nper=nper, etype=np.asarray(etypes, dtype=object)
        )

        parts = {k: np.asarray(v, dtype=np.int64) for k, v in part_map.items()}

        # Surfaces padded
        surfaces: dict[str, SurfaceArray] = {}
        for name, lst in surf_map.items():
            kk = max((len(a) for a in lst), default=0)
            F = len(lst)
            faces = -np.ones((F, kk), dtype=np.int64)
            nps = np.zeros((F,), dtype=np.int64)
            for i, a in enumerate(lst):
                faces[i, : a.size] = a
                nps[i] = a.size
            surfaces[name] = SurfaceArray(faces=faces, nper=nps)

        nodesets = {
            k: np.fromiter(sorted(v), dtype=np.int64) for k, v in nset_map.items()
        }

        return cls(
            nodes=nodes,
            elements=elements,
            parts=parts,
            surfaces=surfaces,
            nodesets=nodesets,
        )

    @classmethod
    def from_xplt(cls, xplt_reader: xplt) -> Mesh:
        """Create from an ``xplt`` reader that already parsed the mesh.

        Args
        ----
        xplt_reader
            Reader whose ``mesh`` attribute is an instance of :class:`Mesh`.
            This method clones that mesh into a detached copy so later edits
            do not mutate the reader's internal buffers.

        Returns
        -------
        Mesh
            New instance.
        """
        xm = xplt_reader.mesh

        if isinstance(xm, Mesh):
            return cls(
                nodes=NodeArray(np.array(xm.nodes.xyz, copy=True)),
                elements=ElementArray(
                    conn=np.array(xm.elements.conn, copy=True),
                    nper=np.array(xm.elements.nper, copy=True),
                    etype=np.array(xm.elements.etype, copy=True),
                ),
                parts={
                    name: np.array(ids, dtype=np.int64, copy=True)
                    for name, ids in xm.parts.items()
                },
                surfaces={
                    name: SurfaceArray(
                        faces=np.array(surf.faces, copy=True),
                        nper=np.array(surf.nper, copy=True),
                    )
                    for name, surf in xm.surfaces.items()
                },
                nodesets={
                    name: np.array(ids, dtype=np.int64, copy=True)
                    for name, ids in xm.nodesets.items()
                },
            )

        raise TypeError("xplt.mesh is not compatible with Mesh.from_xplt")
