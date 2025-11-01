"""Mesh container for interFEBio.

This module defines small data classes to store nodes, elements, and surfaces.
It also provides helpers to export FEBio XML blocks and to build a mesh from
Gmsh ``.msh`` files or from an already parsed ``xplt`` reader.

"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, TypedDict

import numpy as np

if TYPE_CHECKING:
    from interFEBio.XPLT import xplt


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
        """Return the number of nodes."""
        return int(self.xyz.shape[0])

    def __getitem__(self, idx: slice | int | np.ndarray | list[int]) -> NodeArray:
        """Return a sliced view.

        The returned object is a new ``NodeArray`` pointing to a view of
        the original array when possible.
        """
        return NodeArray(self.xyz[idx])

    def take(self, ids: np.ndarray | list[int]) -> NodeArray:
        """Return a new node table with the given node ids.

        Parameters
        ----------
        ids
            0-based node ids to select.
        """
        return NodeArray(self.xyz[np.asarray(ids, dtype=np.int64)])


@dataclass
class ElementArray:
    """Mixed-element connectivity with padding.

    Elements live in a 2D integer array with ``-1`` padding on the right.
    The valid size of each row is stored in ``nper``. This keeps slicing
    and vectorized operations simple while still allowing mixed arity.

    Attributes
    ----------
    conn
        Array of shape ``(E, Kmax)`` with 0-based node ids. Unused slots are ``-1``.
    nper
        Array of shape ``(E,)`` with the valid count per row.
    etype
        Array of shape ``(E,)`` with labels like ``'hex8'`` or ``'tet4'``.
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
        """Return the number of elements."""
        return int(self.conn.shape[0])

    def __getitem__(self, idx: slice | int | np.ndarray | list[int]) -> ElementArray:
        """Return a sliced view."""
        return ElementArray(self.conn[idx], self.nper[idx], self.etype[idx])

    def nodes_of(self, ei: int) -> np.ndarray:
        """Return node ids for a single element.

        Parameters
        ----------
        ei
            Element row index.

        Returns
        -------
        Array of length ``nper[ei]`` with node ids.
        """
        k = self.nper[ei]
        return self.conn[ei, :k]

    def unique_nodes(self) -> np.ndarray:
        """Return sorted unique node ids used by the selection.

        Empty selections return an empty array with ``int64`` dtype.
        """
        if len(self) == 0:
            return np.empty((0,), dtype=np.int64)
        valid = self._mask_valid()
        return np.unique(self.conn[valid])

    def as_ragged(self) -> list[np.ndarray]:
        """Return a Python list with one 1D array of node ids per element.

        Useful when downstream code expects per-element lists.
        """
        return [self.nodes_of(i) for i in range(len(self))]

    def _mask_valid(self) -> np.ndarray:
        """Return a boolean mask of the same shape as ``conn``.

        True marks a valid node slot. False marks a padded slot.
        """
        counts = np.repeat(self.nper[:, None], self.conn.shape[1], axis=1)
        idx = np.repeat(np.arange(self.conn.shape[1])[None, :], len(self), axis=0)
        return idx < counts


@dataclass
class SurfaceArray:
    """Facet connectivity with padding.

    Facets are stored like elements: a padded 2D array plus an ``nper`` vector.

    Attributes
    ----------
    faces
        Array of shape ``(F, Kmax)`` with 0-based node ids. Unused slots are ``-1``.
    nper
        Array of shape ``(F,)`` with valid count per facet.
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
        """Return the number of facets."""
        return int(self.faces.shape[0])

    def __getitem__(self, idx: slice | int | np.ndarray | list[int]) -> SurfaceArray:
        """Return a sliced view."""
        return SurfaceArray(self.faces[idx], self.nper[idx])

    def nodes_of(self, fi: int) -> np.ndarray:
        """Return node ids for a single facet.

        Parameters
        ----------
        fi
            Facet row index.

        Returns
        -------
        Array of length ``nper[fi]`` with node ids.
        """
        k = self.nper[fi]
        return self.faces[fi, :k]

    def unique_nodes(self) -> np.ndarray:
        """Return sorted unique node ids used by the selection.

        Empty selections return an empty array with ``int64`` dtype.
        """
        if len(self) == 0:
            return np.empty((0,), dtype=np.int64)
        valid = self._mask_valid()
        return np.unique(self.faces[valid])

    def _mask_valid(self) -> np.ndarray:
        """Return a boolean mask of the same shape as ``faces``."""
        counts = np.repeat(self.nper[:, None], self.faces.shape[1], axis=1)
        idx = np.repeat(np.arange(self.faces.shape[1])[None, :], len(self), axis=0)
        return idx < counts


class FebNodeCache(TypedDict):
    """Cached node numbering for FEBio XML writers.

    Keys
    ----
    object_name
        Name used when no parts exist.
    part_nodes
        List of tuples ``(part_name, node_ids_sorted)``.
    part_map
        Mapping ``part_name -> {old_node_id: new_node_id}``.
    part_node_sets
        Mapping ``part_name -> set(old_node_id)`` for quick membership checks.
    global_map
        Mapping ``old_node_id -> new_node_id`` across all parts.
    max_node_id
        Highest assigned node id in the cache.
    """

    object_name: str
    part_nodes: list[tuple[str, np.ndarray]]
    part_map: dict[str, dict[int, int]]
    part_node_sets: dict[str, set[int]]
    global_map: dict[int, int]
    max_node_id: int


class Mesh:
    """Unified mesh container.

    Holds node, element, surface, and set tables. Keeps each concern in its own
    class. Provides name-first access and FEBio XML builders.

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
        """Initialize the mesh.

        Parameters
        ----------
        nodes
            Node table.
        elements
            Element table.
        parts
            Optional mapping ``name -> element indices``.
        surfaces
            Optional mapping ``name -> SurfaceArray``.
        nodesets
            Optional mapping ``name -> node ids``.
        """
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
        self._feb_node_cache: FebNodeCache | None = None

    # -------------- simple queries --------------

    @property
    def nelems(self) -> int:
        """Return the number of elements in the mesh."""
        return len(self.elements)

    @property
    def nnodes(self) -> int:
        """Return the number of nodes in the mesh."""
        return len(self.nodes)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the axis-aligned bounding box.

        Returns
        -------
        A tuple ``(min_xyz, max_xyz)`` with two arrays of shape ``(3,)``.
        """
        return self.nodes.xyz.min(0), self.nodes.xyz.max(0)

    # -------------- name-first views --------------

    def getDomain(self, name: str) -> ElementArray:
        """Return a slice of ``elements`` for a named part.

        Parameters
        ----------
        name
            Part name.

        Returns
        -------
        Slice view of ``elements`` for the requested domain.

        Raises
        ------
        KeyError
            If the part name does not exist.
        """
        return self.elements[self.parts[name]]

    def getSurface(self, name: str) -> SurfaceArray:
        """Return a surface by name.

        Parameters
        ----------
        name
            Surface name.

        Returns
        -------
        Surface table.

        Raises
        ------
        KeyError
            If the surface name does not exist.
        """
        return self.surfaces[name]

    def getNodeset(self, name: str) -> NodeArray:
        """Return a nodeset by name as a node slice.

        Parameters
        ----------
        name
            Nodeset name.

        Returns
        -------
        Node slice view.

        Raises
        ------
        KeyError
            If the nodeset name does not exist.
        """
        return self.nodes.take(self.nodesets[name])

    # -------------- FEBio XML writers --------------

    def to_feb_nodes_xml(self, object_name: str = "Object1") -> list[ET.Element]:
        """Build FEBio ``<Nodes>`` elements grouped by part.

        Node ids are re-numbered per part in ascending order of original ids.
        If no parts exist, a single group named ``object_name`` is created.

        Parameters
        ----------
        object_name
            Name used when no parts are present.

        Returns
        -------
        List of ``<Nodes>`` elements. Order follows the internal part order.

        Notes
        -----
        Node ids in XML are 1-based. Coordinates are written as comma-separated
        values ``x,y,z``.
        """
        cache = self._build_feb_node_cache(object_name)
        nodes_xml: list[ET.Element] = []
        part_nodes: list[tuple[str, np.ndarray]] = cache["part_nodes"]
        part_map: dict[str, dict[int, int]] = cache["part_map"]

        for name, node_ids in part_nodes:
            root = ET.Element("Nodes", name=name)
            mapping = part_map[name]
            for nid in node_ids:
                nid_int = int(nid)
                new_id = mapping[nid_int]
                node_el = ET.Element("node", id=str(new_id))
                node_el.text = ",".join(map(str, self.nodes.xyz[nid_int]))
                root.append(node_el)
            nodes_xml.append(root)
        return nodes_xml

    def to_feb_elements_xml(self) -> list[ET.Element]:
        """Build one FEBio ``<Elements>`` element per part.

        Returns
        -------
        List of ``<Elements>`` elements.

        Notes
        -----
        Element ids in XML are 1-based and follow the original element order.
        Element type is taken as the first unique ``etype`` in the part.
        Mixed types in one part are not expanded.
        """
        cache = self._ensure_feb_node_cache()
        part_map: dict[str, dict[int, int]] = cache["part_map"]

        out: list[ET.Element] = []
        for pname, eidx in self.parts.items():
            part = self.elements[eidx]
            # Decide type label: if mixed, use first
            etype = str(np.unique(part.etype)[0]) if len(part) else "unknown"
            el = ET.Element("Elements", type=etype, name=pname)
            node_id_map = part_map.get(pname)
            if node_id_map is None:
                raise KeyError(f"No node id map available for part '{pname}'")
            for local_idx, global_idx in enumerate(eidx):
                nodes = [
                    node_id_map[int(nid)] for nid in part.nodes_of(local_idx).tolist()
                ]
                e_el = ET.Element("elem", id=str(int(global_idx) + 1))
                e_el.text = ",".join(map(str, nodes))
                el.append(e_el)
            out.append(el)
        return out

    def to_feb_surfaces_xml(self) -> list[ET.Element]:
        """Build one FEBio ``<Surface>`` per named surface.

        Returns
        -------
        List of ``<Surface>`` elements.

        Notes
        -----
        Facet tags are chosen by node count: ``line2``, ``tri3``, or ``quad4``.
        Fallback tag is ``facet`` when the count is unknown.
        """
        cache = self._ensure_feb_node_cache()
        part_map: dict[str, dict[int, int]] = cache["part_map"]
        part_node_sets: dict[str, set[int]] = cache["part_node_sets"]
        global_map: dict[int, int] = cache["global_map"]

        out: list[ET.Element] = []
        for sname, surf in self.surfaces.items():
            el = ET.Element("Surface", name=sname)
            for i in range(len(surf)):
                nn = surf.nper[i]
                etype = {2: "line2", 3: "tri3", 4: "quad4"}.get(int(nn), "facet")
                f_el = ET.Element(etype, id=str(i + 1))
                node_list = [int(nid) for nid in surf.nodes_of(i).tolist()]
                mapped = None
                for pname, node_set in part_node_sets.items():
                    if all(n in node_set for n in node_list):
                        mapping = part_map[pname]
                        mapped = [mapping[n] for n in node_list]
                        break
                if mapped is None:
                    try:
                        mapped = [global_map[n] for n in node_list]
                    except KeyError as exc:
                        raise KeyError(
                            f"Surface '{sname}' references unknown node id {int(node_list[0])}"
                        ) from exc
                f_el.text = ",".join(map(str, mapped))
                el.append(f_el)
            out.append(el)
        return out

    def to_feb_nodesets_xml(self) -> list[ET.Element]:
        """Build FEBio ``<NodeSet>`` elements.

        Returns
        -------
        List of ``<NodeSet>`` elements.

        Notes
        -----
        Node ids are re-numbered using the same rules as for nodes and surfaces.
        """
        cache = self._ensure_feb_node_cache()
        part_map: dict[str, dict[int, int]] = cache["part_map"]
        part_node_sets: dict[str, set[int]] = cache["part_node_sets"]
        global_map: dict[int, int] = cache["global_map"]

        out: list[ET.Element] = []
        for nname, nids in self.nodesets.items():
            el = ET.Element("NodeSet", name=nname)
            node_list = [int(nid) for nid in nids.tolist()]
            mapped = None
            for pname, node_set in part_node_sets.items():
                if all(n in node_set for n in node_list):
                    mapping = part_map[pname]
                    mapped = [mapping[n] for n in node_list]
                    break
            if mapped is None:
                try:
                    mapped = [global_map[n] for n in node_list]
                except KeyError as exc:
                    raise KeyError(
                        f"NodeSet '{nname}' references unknown node id {int(node_list[0])}"
                    ) from exc
            el.text = ",".join(map(str, mapped))
            out.append(el)
        return out

    def _build_feb_node_cache(self, object_name: str) -> FebNodeCache:
        """Compute and store FEBio node numbering.

        Rules
        -----
        - If parts exist, nodes are grouped by part. Each group gets 1-based,
          contiguous ids in ascending order of original ids.
        - If no parts exist, all nodes are grouped under ``object_name``.
        - ``global_map`` mirrors the per-part numbering so surfaces and sets
          can map node ids even if they span parts.

        Parameters
        ----------
        object_name
            Name used when no parts are present.

        Returns
        -------
        Cache dictionary used by XML writers.
        """
        part_nodes: list[tuple[str, np.ndarray]] = []
        part_map: dict[str, dict[int, int]] = {}
        part_node_sets: dict[str, set[int]] = {}
        global_map: dict[int, int] = {}

        counter = 1
        if self.parts:
            items = list(self.parts.items())
            for pname, eidx in items:
                part = self.elements[eidx]
                node_ids = np.asarray(part.unique_nodes(), dtype=np.int64)
                node_ids.sort()
                local_map: dict[int, int] = {}
                node_set: set[int] = set()
                for nid in node_ids.tolist():
                    nid_int = int(nid)
                    new_id = counter
                    counter += 1
                    local_map[nid_int] = new_id
                    global_map[nid_int] = new_id
                    node_set.add(nid_int)
                part_nodes.append((pname, node_ids))
                part_map[pname] = local_map
                part_node_sets[pname] = node_set
        else:
            node_ids = np.arange(self.nnodes, dtype=np.int64)
            local_map = {int(nid): counter + idx for idx, nid in enumerate(node_ids)}
            counter += len(node_ids)
            part_nodes.append((object_name, node_ids))
            part_map[object_name] = local_map
            part_node_sets[object_name] = set(local_map.keys())
            global_map.update(local_map)

        cache: FebNodeCache = {
            "object_name": object_name,
            "part_nodes": part_nodes,
            "part_map": part_map,
            "part_node_sets": part_node_sets,
            "global_map": global_map,
            "max_node_id": counter - 1,
        }
        self._feb_node_cache = cache
        return cache

    def _ensure_feb_node_cache(self, object_name: str | None = None) -> FebNodeCache:
        """Return cached FEBio node numbering, building it if needed.

        Parameters
        ----------
        object_name
            When given and different from the cache, the cache is rebuilt.

        Returns
        -------
        Cache dictionary used by XML writers.
        """
        cache = self._feb_node_cache
        if cache is None:
            name = object_name or "Object1"
            return self._build_feb_node_cache(name)
        if object_name is not None and cache["object_name"] != object_name:
            return self._build_feb_node_cache(object_name)
        return cache

    # -------------- constructors from sources --------------

    @classmethod
    def from_gmsh_msh(
        cls,
        path: str,
        include: Sequence[str] | None = None,
        scale: Sequence[float] | np.ndarray = (1.0, 1.0, 1.0),
        quiet: bool = False,
    ) -> Mesh:
        """Create a mesh from a Gmsh 2.0/2.2 ASCII ``.msh`` file.

        The reader:
        - Supports physical groups for volumes, surfaces, and lines.
        - Builds parts from volume groups. Optionally filters by ``include``.
        - Builds surfaces and nodesets from surface/line groups.
        - Reindexes node ids to 0-based and preserves original ordering.
        - Pads connectivity to a common width for simple slicing.

        Parameters
        ----------
        path
            Path to the ``.msh`` file.
        include
            Physical names to include as parts. When ``None``, include all volumes.
        scale
            Per-axis scale for node coordinates. Applied as ``(sx, sy, sz)``.
        quiet
            When ``True``, suppress warnings about unsupported types.

        Returns
        -------
        New ``Mesh`` instance.

        Raises
        ------
        SystemExit
            When the file uses an unsupported Gmsh format.
        ValueError
            When the file has inconsistent sections.

        Notes
        -----
        Only Gmsh ASCII formats 2.0 and 2.2 are supported. Quadratic types are
        kept as-is but not expanded. Unknown element types are skipped.
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
        """Create a mesh from an ``xplt`` reader that already parsed a mesh.

        Parameters
        ----------
        xplt_reader
            Reader whose ``mesh`` attribute is an instance compatible with ``Mesh``.

        Returns
        -------
        Detached copy of the reader mesh.

        Raises
        ------
        TypeError
            When ``xplt_reader.mesh`` is not compatible with this class.
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
