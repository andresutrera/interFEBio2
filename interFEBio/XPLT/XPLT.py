"""FEBio .xplt reader with regional FMT_NODE support and sliceable result views.

Overview
========
This module reads FEBio ``.xplt`` files and exposes results through *views*.
A view is a small object that lets you slice results by time, by region, by
item (element or face), by node inside an item, and by component. Views do
not copy data unless needed. They return NumPy arrays on demand.

Workflow
--------
1) Header and dictionary
   - The reader opens the file and checks magic, version, and compression.
   - The dictionary is parsed. Each variable gets a data type and a storage
     format (FMT_NODE, FMT_ITEM, FMT_MULT, FMT_REGION). The dictionary also
     sets where the variable lives (node, elem, face).

2) Mesh
   - Nodes, domains (parts), surfaces, and nodesets are parsed.
   - Connectivity is padded to a fixed width so slicing stays simple.
   - A :class:`Mesh` object is built with separate node, element, and surface
     tables. Domain and surface names are preserved.

3) States
   - You can read every state with ``readAllStates()`` or a list of steps
     with ``readSteps([ ... ])``.
   - Each state holds time and a block of result streams:
     node, element, and face.
   - For each stream the reader detects the format and region id then stores
     per-state data in scratch buffers. Missing data is recorded as ``None``.

4) Finalization
   - After reading, per-variable buffers are converted into *views* and
     attached to a :class:`Results` container under ``xplt.results``.
   - No interpolation is done. Shapes are left as written.

Views concept
=============
A *view* lets you request a rectangular block from a result without knowing
how it is stored inside the reader. The API is the same across formats:

- ``time(idx)`` picks states by index or slice.
- ``comp(idx)`` picks components by index, slice, or name (when available).
- Region-aware views support:
  - ``region(name)`` or ``domain(name)`` for parts,
  - ``surface(name)`` for faces,
  - ``regions()`` to list names.
- Indexers for items and nodes:
  - ``items(ids)`` or aliases ``elems(ids)`` and ``faces(ids)``.
  - ``enodes(ids)`` for per-element-node data.
  - ``nodes(ids)`` for nodal data.
- Shorthand slicing with ``__getitem__`` is available on some views.

Selectors accept:
- integers,
- Python slices,
- lists or NumPy index arrays,
- the string token ``":"`` as a shorthand for "all".

Returned array shapes follow the view's ``dims()`` description.

View types
----------
- :class:`NodeResultView` (FMT_NODE global):
  dims = (time, node, component).

- :class:`NodeRegionResultView` (FMT_NODE per-domain):
  dims = (time, node_in_region, component).
  First choose a region with ``region(name)`` or index via ``[t, node_sel, comp]``
  after selecting a single region.

- :class:`ItemResultView` (FMT_ITEM for domains or surfaces):
  dims = (time, item, component). Items are elements for domains and faces for
  surfaces.

- :class:`MultResultView` (FMT_MULT per item per element-node):
  dims = (time, item, enode, component). Padding with NaN is used when an item
  has fewer nodes than the maximum in the region.

- :class:`RegionResultView` (FMT_REGION one vector per region):
  dims = (time, component). Use ``region(name)`` to select a single region.

Component names
---------------
When a variable defines named components the following tokens can be used:

- VEC3F: ``"x" "y" "z"``
- MAT3FD: ``"xx" "yy" "zz"``
- MAT3FS: ``"xx" "yy" "zz" "xy" "yz" "xz"``
- MAT3F: full 3x3 in row-major tokens
- TENS4FS: two-token pairs in Voigt-6 like ``"xxyy"``

If names are not available, only integer or slice component selectors work.

Usage examples
==============
Read all states and get displacement at a nodeset:

>>> xp = xplt("run.xplt")
>>> xp.readAllStates()
>>> r = xp.results
>>> U = r["displacement"]                  # NodeResultView or NodeRegionResultView
>>> U0 = U.time(0).comp("x")               # first time, x-component, all nodes
>>> Uset = U.nodes(xp.mesh.nodesets["base"]).comp("z")

Direct slicing with the index operator on node results:

>>> Uxz = r["displacement"][0, :, "x"]     # time 0, all nodes, x-comp

Per-domain nodal results (FMT_NODE per region):

>>> Ur = r["displacement"].region("arteria")
>>> Ur_vals = Ur.time(":").nodes(slice(0, 10)).comp("y")

Element item results:

>>> S = r["von Mises"].domain("arteria")   # ItemResultView
>>> S_last = S.time(-1).items([0, 5, 9]).comp(":")

Per-element-node results:

>>> Q = r["strain"].domain("arteria")      # MultResultView
>>> Qpick = Q.time(0).items(0).enodes(":").comp("xx")

Region vector results:

>>> R = r["domain volume"].domain("arteria")  # RegionResultView
>>> R_all_t = R.time(":").comp(":")

Missing data
------------
If a variable is missing for a given time or region, the view returns arrays
filled with NaN for that block. Shapes are preserved.

All views convert outputs to float32 to keep memory usage low.

Implementation notes
--------------------
- Domain and surface names are taken from the file when available.
- Node ids in the file may be 1-based or 0-based. The reader normalizes them.
- Connectivity and facet lists are right-padded with ``-1`` for uniform width.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from os import SEEK_SET
from typing import Dict, Iterable, List

import numpy as np

from interFEBio.Mesh.Mesh import ElementArray, Mesh, NodeArray, SurfaceArray

from .BinaryReader import BinaryReader
from .Enums import (
    Elem_Type,
    FEDataDim,
    FEDataType,
    Storage_Fmt,
    nodesPerElementClass,
)

# --------------------------- helpers ---------------------------

Index = int | slice | Iterable[int] | np.ndarray | None


def _norm_node_ids(ids: np.ndarray, N: int) -> np.ndarray:
    """Normalize node ids to 0-based contiguous indices.

    Accepts arrays that may be 0-based or 1-based. Verifies range and
    converts to 0-based if needed.

    Returns the normalized array with the same shape.
    """
    a = np.asarray(ids, dtype=np.int64, order="C")
    if a.size == 0:
        return a
    valid = a >= 0
    if not np.any(valid):
        return a
    vmax = int(a[valid].max())
    if vmax == N:
        a[valid] -= 1
    elif vmax > N:
        raise ValueError(f"node id {vmax} exceeds node count {N}")
    if np.any(a[valid] < 0) or np.any(a[valid] >= N):
        raise ValueError("normalized node ids out of range")
    return a


class _FieldMeta:
    """Lightweight metadata for a result variable.

    Holds name, storage format, base data type, and component count.
    """

    __slots__ = ("name", "fmt", "dtype", "ncomp")

    def __init__(self, name: str, fmt: Storage_Fmt, dtype: FEDataType):
        self.name = name
        self.fmt = fmt
        self.dtype = dtype
        self.ncomp = int(FEDataDim[dtype.name].value)

    def __repr__(self) -> str:
        return f"_FieldMeta(name={self.name!r}, fmt={self.fmt.name}, dtype={self.dtype.name}, ncomp={self.ncomp})"

    __str__ = __repr__


_VEC3_ORDER = ("x", "y", "z")
_MAT3FD_ORDER = ("xx", "yy", "zz")
_MAT3FS_ORDER = ("xx", "yy", "zz", "xy", "yz", "xz")
_MAT3F_ORDER = ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")
_VOIGT6 = {"xx": 0, "yy": 1, "zz": 2, "yz": 3, "xz": 4, "xy": 5}


def _normalize_comp_token(s: str) -> str:
    """Normalize a component token.

    Lowercases and strips spaces so user input is tolerant of formatting.
    """
    return s.strip().lower().replace(" ", "")


def _comp_names_for_dtype(dtype: FEDataType) -> tuple[str, ...] | None:
    """Return valid component names for known data types or None."""
    if dtype == FEDataType.VEC3F:
        return _VEC3_ORDER
    if dtype == FEDataType.MAT3FD:
        return _MAT3FD_ORDER
    if dtype == FEDataType.MAT3FS:
        return _MAT3FS_ORDER
    if dtype == FEDataType.MAT3F:
        return _MAT3F_ORDER
    return None


def _tens4fs_pair_index(p: str) -> int:
    """Map a 4th-order Voigt pair token like 'xxyy' to an index.

    Uses a symmetric pair mapping with Voigt-6 order.
    """
    p = _normalize_comp_token(p)
    if len(p) != 4:
        raise KeyError(f"invalid 4th-order token '{p}'")
    a, b = p[:2], p[2:]
    if a not in _VOIGT6 or b not in _VOIGT6:
        raise KeyError(f"invalid pair in '{p}'")
    i, j = _VOIGT6[a], _VOIGT6[b]
    if i > j:
        i, j = j, i
    return i + (j * (j + 1)) // 2


def _comp_spec_to_index(meta: _FieldMeta, spec: Index | str | Iterable[str]) -> Index:
    """Convert a component selector to an integer, slice, array, or None.

    Accepts:
    - integer index
    - Python slice
    - NumPy index array
    - string name like 'x' or 'xx'
    - list of string names
    - the string token ':' for all components
    """
    if spec is None:
        return None
    if isinstance(spec, (int, slice, np.ndarray)):
        return spec
    if isinstance(spec, str):
        s = _normalize_comp_token(spec)
        if s == ":":
            return slice(None)
        names = _comp_names_for_dtype(meta.dtype)
        if names is not None:
            try:
                return names.index(s)
            except ValueError as e:
                raise KeyError(
                    f"component '{spec}' not valid for {meta.dtype.name}. Valid: {names}"
                ) from e
        if meta.dtype == FEDataType.TENS4FS:
            return _tens4fs_pair_index(s)
        raise KeyError(f"component strings not supported for {meta.dtype.name}")
    try:
        lst = list(spec)  # type: ignore[arg-type]
    except TypeError as e:
        raise TypeError("unsupported component selector") from e
    out: List[int] = []
    for item in lst:
        if not isinstance(item, str):
            raise TypeError("component list must be strings")
        idx_item = _comp_spec_to_index(meta, item)
        if isinstance(idx_item, int):
            out.append(idx_item)
        else:
            raise TypeError("component list does not accept slices")
    return np.asarray(out, dtype=np.int64)


def _as_index_list(sel: Index, T: int) -> List[int]:
    """Normalize a selector to a list of integer indices."""
    if sel is None:
        return list(range(T))
    if isinstance(sel, int):
        return [sel]
    if isinstance(sel, slice):
        return list(range(T))[sel]
    if isinstance(sel, np.ndarray) and sel.dtype == bool:
        return list(np.nonzero(sel)[0].tolist())
    return list(sel)  # type: ignore[arg-type]


def _shape_from_slice(n: int, s: Index) -> int:
    """Compute the resulting length when a selector is applied."""
    if s is None:
        return n
    if isinstance(s, int):
        return 1
    if isinstance(s, slice):
        start, stop, step = s.indices(n)
        return max(0, (stop - start + (step - 1)) // step)
    return len(list(s))  # type: ignore[arg-type]


# --------------------------- views ---------------------------


class _BaseView:
    """Common behavior for result views.

    Stores metadata and selection state for time and component axes.
    """

    __slots__ = ("meta", "_times", "_comp_idx", "_t_idx")

    def __init__(self, meta: _FieldMeta, times: np.ndarray):
        self.meta = meta
        self._times = np.asarray(times, float)
        self._comp_idx: Index = None
        self._t_idx: Index = None

    def time(self, idx: Index | str = None):
        """Select time steps.

        Accepts integer, slice, list/array, boolean mask, or ':' string.
        Returns the view itself to allow chaining.
        """
        if idx == ":":
            idx = slice(None)
        self._t_idx = idx  # type: ignore[assignment]
        return self

    def comp(self, idx: Index | str | Iterable[str]):
        """Select components and compute the array.

        Accepts integer, slice, list/array, or component names.
        Returns a NumPy array with the current selections applied.
        """
        self._comp_idx = _comp_spec_to_index(self.meta, idx)
        return self.eval()

    def dims(self) -> tuple[str, ...]:
        """Describe array axes in order."""
        raise NotImplementedError

    def eval(self) -> np.ndarray:
        """Build the array for the current selection."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of time steps."""
        return int(self._times.shape[0])


class NodeResultView(_BaseView):
    """Global nodal results (FMT_NODE).

    dims() -> ('time', 'node', 'component')

    Selection
    ---------
    - time(...)
    - nodes(...)
    - comp(...)

    Shorthand
    ---------
    - view[t, nodes, comp]
      where any of the three can be ':' for all.
    """

    __slots__ = ("_per_t", "_mesh", "_node_idx")

    def __init__(
        self,
        meta: _FieldMeta,
        times: np.ndarray,
        per_t: List[np.ndarray | None],
        mesh: Mesh,
    ):
        super().__init__(meta, times)
        self._per_t = per_t
        self._mesh = mesh
        self._node_idx: Index = None

    def nodes(self, ids: Index):
        """Select node rows by indices, slices, masks, or lists."""
        self._node_idx = ids
        return self

    def nodeset(self, name: str):
        """Select by nodeset name from the mesh."""
        self._node_idx = self._mesh.nodesets[name]
        return self

    def dims(self) -> tuple[str, ...]:
        return ("time", "node", "component")

    def eval(self) -> np.ndarray:
        """Return an array with the current time, node, and component selection."""
        T_sel = _as_index_list(self._t_idx, len(self))
        N0 = next((a.shape[0] for a in self._per_t if a is not None), 0)
        C0 = self.meta.ncomp
        N_sel = _shape_from_slice(N0, self._node_idx)
        C_sel = _shape_from_slice(C0, self._comp_idx)

        def _one(k: int) -> np.ndarray:
            a = self._per_t[k]
            if a is None:
                return np.full((N_sel, C_sel), np.nan, dtype=np.float32)
            out = a
            if self._node_idx is not None:
                out = out[self._node_idx, :]  # type: ignore[index]
            if self._comp_idx is not None:
                out = out[..., self._comp_idx]  # type: ignore[index]
            out = np.asarray(out, dtype=np.float32)
            if out.ndim == 1:
                out = out.reshape(1, -1) if N_sel == 1 else out.reshape(-1, 1)
            return out

        if len(T_sel) == 1 and isinstance(self._t_idx, int):
            return _one(T_sel[0])
        return np.stack([_one(k) for k in T_sel], axis=0)

    def __getitem__(self, key):
        """Shorthand selection: [time, nodes, comp].

        Any position can be ':' as a string for all. Missing positions default
        to ':'.
        """
        if not isinstance(key, tuple):
            t = key
            if isinstance(t, str) and t == ":":
                t = slice(None)
            self._t_idx = t  # int | slice | np.ndarray | Iterable[int] | None
            self._node_idx = None
            self._comp_idx = None
            return self.eval()

        # normalize 3-tuple
        t, n, c = (key + (slice(None),) * 3)[:3]

        # only convert ":" when it is a string token
        if isinstance(t, str) and t == ":":
            t = slice(None)
        if isinstance(n, str) and n == ":":
            n = slice(None)
        if isinstance(c, str):
            c = slice(None) if c == ":" else _comp_spec_to_index(self.meta, c)

        # store raw selectors; numpy arrays are supported
        self._t_idx = t  # int | slice | np.ndarray[bool|int] | Iterable[int]
        self._node_idx = n  # int | slice | np.ndarray[bool|int] | Iterable[int]
        self._comp_idx = c  # int | slice | np.ndarray[bool|int] | Iterable[int] | None
        return self.eval()

    def __repr__(self) -> str:
        N = next((a.shape[0] for a in self._per_t if a is not None), 0)
        C = next((a.shape[1] for a in self._per_t if a is not None), self.meta.ncomp)
        missing = sum(1 for a in self._per_t if a is None)
        return f"NodeResultView(name={self.meta.name!r}, T={len(self)}, N={N}, C={C}, missing={missing})"

    __str__ = __repr__


class NodeRegionResultView(_BaseView):
    """Per-domain nodal results (FMT_NODE split by region).

    dims() -> ('time', 'node_in_region', 'component')

    You must first select a single region with ``region(name)`` when using
    index-based shorthand selection.
    """

    __slots__ = ("_per_name", "_node_idx", "_region_nodes")

    def __init__(
        self,
        meta: _FieldMeta,
        times: np.ndarray,
        per_name: Dict[str, List[np.ndarray | None]],
        region_nodes: Dict[str, np.ndarray],
    ):
        super().__init__(meta, times)
        self._per_name = per_name
        self._region_nodes = region_nodes
        self._node_idx: Index = None

    def regions(self) -> List[str]:
        """Return region names available for this variable."""
        return sorted(self._per_name.keys())

    domains = regions

    def region(self, name: str):
        """Return a new view restricted to a single region."""
        if name not in self._per_name:
            raise KeyError(name)
        return NodeRegionResultView(
            self.meta,
            self._times,
            {name: self._per_name[name]},
            {name: self._region_nodes[name]},
        )

    domain = region

    def region_nodes(self) -> np.ndarray:
        """Return the node ids of the selected region.

        Only valid when the view holds a single region.
        """
        if len(self._region_nodes) != 1:
            raise ValueError("multiple regions present; select region() first")
        return next(iter(self._region_nodes.values()))

    def nodes(self, ids: Index):
        """Select nodes by indices relative to the region list."""
        self._node_idx = ids
        return self

    def dims(self) -> tuple[str, ...]:
        return ("time", "node_in_region", "component")

    def _pick_per_t(self) -> List[np.ndarray | None]:
        if len(self._per_name) != 1:
            raise ValueError("multiple regions present; select region() first")
        return next(iter(self._per_name.values()))

    def eval(self, *, region: str | None = None) -> np.ndarray:
        """Return an array with the current selection."""
        per_t = self._pick_per_t() if region is None else self._per_name[region]
        T_sel = _as_index_list(self._t_idx, len(self))
        N0 = next((a.shape[0] for a in per_t if a is not None), 0)
        C0 = self.meta.ncomp
        N_sel = _shape_from_slice(N0, self._node_idx)
        C_sel = _shape_from_slice(C0, self._comp_idx)

        def _one(k: int) -> np.ndarray:
            a = per_t[k]
            if a is None:
                return np.full((N_sel, C_sel), np.nan, dtype=np.float32)
            out = a
            if self._node_idx is not None:
                out = out[self._node_idx, :]  # type: ignore[index]
            if self._comp_idx is not None:
                out = out[..., self._comp_idx]  # type: ignore[index]
            out = np.asarray(out, dtype=np.float32)
            if out.ndim == 1:
                out = out.reshape(1, -1) if N_sel == 1 else out.reshape(-1, 1)
            return out

        if len(T_sel) == 1 and isinstance(self._t_idx, int):
            return _one(T_sel[0])
        return np.stack([_one(k) for k in T_sel], axis=0)

    def __repr__(self) -> str:
        names = list(self._per_name.keys())
        T = len(self)
        if len(names) == 1:
            name = names[0]
            per_t = self._per_name[name]
            N = next((a.shape[0] for a in per_t if a is not None), 0)
            C = next((a.shape[1] for a in per_t if a is not None), self.meta.ncomp)
            missing = sum(1 for a in per_t if a is None)
            return f"NodeRegionResultView(name={self.meta.name!r}, region={name!r}, T={T}, N={N}, C={C}, missing={missing})"
        parts = []
        for d in sorted(names):
            per_t = self._per_name[d]
            N = next((a.shape[0] for a in per_t if a is not None), 0)
            C = next((a.shape[1] for a in per_t if a is not None), self.meta.ncomp)
            missing = sum(1 for a in per_t if a is None)
            parts.append(f"{d}:N={N},C={C},missing={missing}")
        return (
            f"NodeRegionResultView(name={self.meta.name!r}, T={T}, regions={len(names)} | "
            + "; ".join(parts)
            + ")"
        )

    __str__ = __repr__


class ItemResultView(_BaseView):
    """Per-item results (FMT_ITEM).

    Items are elements for domains and faces for surfaces.

    dims() -> ('time', 'item', 'component')
    """

    __slots__ = ("_per_name", "_item_idx")

    def __init__(
        self,
        meta: _FieldMeta,
        times: np.ndarray,
        per_name: Dict[str, List[np.ndarray | None]],
    ):
        super().__init__(meta, times)
        self._per_name = per_name
        self._item_idx: Index = None

    def regions(self) -> List[str]:
        """Return region names available for this variable."""
        return sorted(self._per_name.keys())

    domains = regions
    surfaces = regions

    def region(self, name: str):
        """Return a new view restricted to a single region."""
        if name not in self._per_name:
            raise KeyError(name)
        return ItemResultView(self.meta, self._times, {name: self._per_name[name]})

    domain = region
    surface = region

    def items(self, ids: Index):
        """Select items by indices, slices, masks, or lists."""
        self._item_idx = ids
        return self

    elems = items
    faces = items

    def dims(self) -> tuple[str, ...]:
        return ("time", "item", "component")

    def _pick_per_t(self) -> List[np.ndarray | None]:
        if len(self._per_name) != 1:
            raise ValueError("multiple regions present; select region() first")
        return next(iter(self._per_name.values()))

    def eval(self, *, region: str | None = None) -> np.ndarray:
        """Return an array with the current selection."""
        per_t = self._pick_per_t() if region is None else self._per_name[region]
        T_sel = _as_index_list(self._t_idx, len(self))
        R0 = next((a.shape[0] for a in per_t if a is not None), 0)
        C0 = self.meta.ncomp
        R_sel = _shape_from_slice(R0, self._item_idx)
        C_sel = _shape_from_slice(C0, self._comp_idx)

        def _one(k: int) -> np.ndarray:
            a = per_t[k]
            if a is None:
                return np.full((R_sel, C_sel), np.nan, dtype=np.float32)
            out = a
            if self._item_idx is not None:
                out = out[self._item_idx, :]  # type: ignore[index]
            if self._comp_idx is not None:
                out = out[..., self._comp_idx]  # type: ignore[index]
            out = np.asarray(out, dtype=np.float32)
            if out.ndim == 1:
                out = out.reshape(1, -1) if R_sel == 1 else out.reshape(-1, 1)
            return out

        if len(T_sel) == 1 and isinstance(self._t_idx, int):
            return _one(T_sel[0])
        return np.stack([_one(k) for k in T_sel], axis=0)

    def __getitem__(self, key):
        """Shorthand selection: [time, item, comp]."""
        _ = self._pick_per_t()
        if not isinstance(key, tuple):
            t = key
            self._t_idx = t if t != ":" else slice(None)
            self._item_idx = None
            self._comp_idx = None
            return self.eval()
        t, r, c = (key + (slice(None),) * 3)[:3]
        if t == ":":
            t = slice(None)
        if r == ":":
            r = slice(None)
        if isinstance(c, str):
            c = _comp_spec_to_index(self.meta, c)
        self._t_idx = t
        self._item_idx = r
        self._comp_idx = c
        return self.eval()

    def __repr__(self) -> str:
        names = list(self._per_name.keys())
        T = len(self)
        if len(names) == 1:
            name = names[0]
            per_t = self._per_name[name]
            R = next((a.shape[0] for a in per_t if a is not None), 0)
            C = next((a.shape[1] for a in per_t if a is not None), self.meta.ncomp)
            missing = sum(1 for a in per_t if a is None)
            return f"ItemResultView(name={self.meta.name!r}, region={name!r}, T={T}, R={R}, C={C}, missing={missing})"
        parts = []
        for d in sorted(names):
            per_t = self._per_name[d]
            R = next((a.shape[0] for a in per_t if a is not None), 0)
            C = next((a.shape[1] for a in per_t if a is not None), self.meta.ncomp)
            missing = sum(1 for a in per_t if a is None)
            parts.append(f"{d}:R={R},C={C},missing={missing}")
        return (
            f"ItemResultView(name={self.meta.name!r}, T={T}, regions={len(names)} | "
            + "; ".join(parts)
            + ")"
        )

    __str__ = __repr__


@dataclass
class MultBlock:
    """Packed data for FMT_MULT.

    data
        3D array with shape (items, max_enodes, components). Cells for missing
        enodes are filled with NaN.
    nper
        Number of valid enodes per item.
    """

    data: np.ndarray  # (R, Kmax, C) float32 with NaN padding
    nper: np.ndarray  # (R,) int64


class MultResultView(_BaseView):
    """Per-item per-element-node results (FMT_MULT).

    dims() -> ('time', 'item', 'enode', 'component')
    """

    __slots__ = ("_per_name", "_item_idx", "_enode_idx")

    def __init__(
        self,
        meta: _FieldMeta,
        times: np.ndarray,
        per_name: Dict[str, List[MultBlock | None]],
    ):
        super().__init__(meta, times)
        self._per_name = per_name
        self._item_idx: Index = None
        self._enode_idx: Index = None

    def regions(self) -> List[str]:
        """Return region names available for this variable."""
        return sorted(self._per_name.keys())

    domains = regions
    surfaces = regions

    def region(self, name: str):
        """Return a new view restricted to a single region."""
        if name not in self._per_name:
            raise KeyError(name)
        return MultResultView(self.meta, self._times, {name: self._per_name[name]})

    domain = region
    surface = region

    def items(self, ids: Index):
        """Select items by indices, slices, masks, or lists."""
        self._item_idx = ids
        return self

    elems = items
    faces = items

    def enodes(self, ids: Index):
        """Select element-node positions by indices or slices."""
        self._enode_idx = ids
        return self

    def dims(self) -> tuple[str, ...]:
        return ("time", "item", "enode", "component")

    def _pick_per_t(self) -> List[MultBlock | None]:
        if len(self._per_name) != 1:
            raise ValueError("multiple regions present; select region() first")
        return next(iter(self._per_name.values()))

    def eval(self, *, region: str | None = None) -> np.ndarray:
        """Return an array with the current selection."""
        per_t = self._pick_per_t() if region is None else self._per_name[region]
        T_sel = _as_index_list(self._t_idx, len(self))
        first = next((mb for mb in per_t if mb is not None), None)
        if first is None:
            C_sel = _shape_from_slice(self.meta.ncomp, self._comp_idx)
            return np.full((len(T_sel), 0, 0, C_sel), np.nan, dtype=np.float32)
        R0, Kmax, C0 = first.data.shape
        R_sel = _shape_from_slice(R0, self._item_idx)
        K_sel = _shape_from_slice(Kmax, self._enode_idx)
        C_sel = _shape_from_slice(C0, self._comp_idx)

        def _one(k: int) -> np.ndarray:
            mb = per_t[k]
            if mb is None:
                return np.full((R_sel, K_sel, C_sel), np.nan, dtype=np.float32)
            out = mb.data
            if self._item_idx is not None:
                out = out[self._item_idx, :, :]  # type: ignore[index]
            if self._enode_idx is not None:
                out = out[:, self._enode_idx, :]  # type: ignore[index]
            if self._comp_idx is not None:
                out = out[..., self._comp_idx]  # type: ignore[index]
            return np.asarray(out, dtype=np.float32)

        if len(T_sel) == 1 and isinstance(self._t_idx, int):
            return _one(T_sel[0])
        return np.stack([_one(k) for k in T_sel], axis=0)

    def __repr__(self) -> str:
        names = list(self._per_name.keys())
        T = len(self)
        if len(names) == 1:
            name = names[0]
            mb = next((b for b in self._per_name[name] if b is not None), None)
            if mb is None:
                return f"MultResultView(name={self.meta.name!r}, region={name!r}, T={T}, R=0, Kmax=0, C={self.meta.ncomp}, missing={T})"
            R, Kmax, C = mb.data.shape
            missing = sum(1 for b in self._per_name[name] if b is None)
            return f"MultResultView(name={self.meta.name!r}, region={name!r}, T={T}, R={R}, Kmax={Kmax}, C={C}, missing={missing})"
        parts = []
        for d in sorted(names):
            mb = next((b for b in self._per_name[d] if b is not None), None)
            if mb is None:
                parts.append(f"{d}:R=0,Kmax=0,C={self.meta.ncomp},missing={len(self)}")
            else:
                R, Kmax, C = mb.data.shape
                missing = sum(1 for b in self._per_name[d] if b is None)
                parts.append(f"{d}:R={R},Kmax={Kmax},C={C},missing={missing}")
        return (
            f"MultResultView(name={self.meta.name!r}, T={T}, regions={len(names)} | "
            + "; ".join(parts)
            + ")"
        )

    __str__ = __repr__


class RegionResultView(_BaseView):
    """Per-region vector results (FMT_REGION).

    dims() -> ('time', 'component')
    """

    __slots__ = ("_per_name",)

    def __init__(
        self,
        meta: _FieldMeta,
        times: np.ndarray,
        per_name: Dict[str, List[np.ndarray | None]],
    ):
        super().__init__(meta, times)
        self._per_name = per_name

    def regions(self) -> List[str]:
        """Return region names available for this variable."""
        return sorted(self._per_name.keys())

    domains = regions
    surfaces = regions

    def region(self, name: str):
        """Return a new view restricted to a single region."""
        if name not in self._per_name:
            raise KeyError(name)
        return RegionResultView(self.meta, self._times, {name: self._per_name[name]})

    domain = region
    surface = region

    def dims(self) -> tuple[str, ...]:
        return ("time", "component")

    def _pick_per_t(self) -> List[np.ndarray | None]:
        if len(self._per_name) != 1:
            raise ValueError("multiple regions present; select region() first")
        return next(iter(self._per_name.values()))

    def eval(self, *, region: str | None = None) -> np.ndarray:
        """Return an array with the current selection."""
        per_t = self._pick_per_t() if region is None else self._per_name[region]
        T_sel = _as_index_list(self._t_idx, len(self))
        C0 = self.meta.ncomp
        C_sel = _shape_from_slice(C0, self._comp_idx)

        def _one(k: int) -> np.ndarray:
            a = per_t[k]
            if a is None:
                return np.full((C_sel,), np.nan, dtype=np.float32)
            out = a
            if self._comp_idx is not None:
                out = out[self._comp_idx]  # type: ignore[index]
            return np.asarray(out, dtype=np.float32).reshape(-1)

        if len(T_sel) == 1 and isinstance(self._t_idx, int):
            return _one(T_sel[0])
        return np.stack([_one(k) for k in T_sel], axis=0)

    def __repr__(self) -> str:
        names = list(self._per_name.keys())
        T = len(self)
        if len(names) == 1:
            name = names[0]
            per_t = self._per_name[name]
            C = next((a.shape[-1] for a in per_t if a is not None), self.meta.ncomp)
            missing = sum(1 for a in per_t if a is None)
            return f"RegionResultView(name={self.meta.name!r}, region={name!r}, T={T}, C={C}, missing={missing})"
        parts = []
        for d in sorted(names):
            per_t = self._per_name[d]
            C = next((a.shape[-1] for a in per_t if a is not None), self.meta.ncomp)
            missing = sum(1 for a in per_t if a is None)
            parts.append(f"{d}:C={C},missing={missing}")
        return (
            f"RegionResultView(name={self.meta.name!r}, T={T}, regions={len(names)} | "
            + "; ".join(parts)
            + ")"
        )

    __str__ = __repr__


# --------------------------- results container ---------------------------


class Results:
    """Container for all result views at all locations.

    Access
    ------
    - ``results[name]`` returns the best-matching view for the variable name.
      It checks in this order:
        node, node_region, elem_item, face_item, elem_mult, face_mult,
        elem_region, face_region.

    Introspection
    -------------
    - ``len(results)`` returns number of time steps.
    - ``results.times()`` returns the time vector.
    """

    def __init__(self, times: Iterable[float]):
        self._times = np.asarray(list(times), float)
        self.node: Dict[str, NodeResultView] = {}  # global nodal
        self.node_region: Dict[str, NodeRegionResultView] = {}  # nodal per-domain
        self.elem_item: Dict[str, ItemResultView] = {}
        self.face_item: Dict[str, ItemResultView] = {}
        self.elem_mult: Dict[str, MultResultView] = {}
        self.face_mult: Dict[str, MultResultView] = {}
        self.elem_region: Dict[str, RegionResultView] = {}
        self.face_region: Dict[str, RegionResultView] = {}
        self._meta: Dict[str, _FieldMeta] = {}

    def __len__(self) -> int:
        """Return number of time steps."""
        return int(self._times.shape[0])

    def times(self) -> np.ndarray:
        """Return the time vector."""
        return self._times

    def register_node_global(
        self, name: str, meta: _FieldMeta, per_t: List[np.ndarray | None], mesh: Mesh
    ):
        """Register a global nodal variable."""
        self._meta[name] = meta
        self.node[name] = NodeResultView(meta, self._times, per_t, mesh)

    def register_node_region(
        self,
        name: str,
        meta: _FieldMeta,
        per_name: Dict[str, List[np.ndarray | None]],
        region_nodes: Dict[str, np.ndarray],
    ):
        """Register a per-domain nodal variable."""
        self._meta[name] = meta
        self.node_region[name] = NodeRegionResultView(
            meta, self._times, per_name, region_nodes
        )

    def register_item(
        self,
        where: str,
        name: str,
        meta: _FieldMeta,
        per_name: Dict[str, List[np.ndarray | None]],
    ):
        """Register an item variable for elements or faces."""
        self._meta[name] = meta
        v = ItemResultView(meta, self._times, per_name)
        if where == "elem":
            self.elem_item[name] = v
        else:
            self.face_item[name] = v

    def register_mult(
        self,
        where: str,
        name: str,
        meta: _FieldMeta,
        per_name: Dict[str, List[MultBlock | None]],
    ):
        """Register a per-element-node variable for elements or faces."""
        self._meta[name] = meta
        v = MultResultView(meta, self._times, per_name)
        if where == "elem":
            self.elem_mult[name] = v
        else:
            self.face_mult[name] = v

    def register_region(
        self,
        where: str,
        name: str,
        meta: _FieldMeta,
        per_name: Dict[str, List[np.ndarray | None]],
    ):
        """Register a per-region vector variable for elements or faces."""
        self._meta[name] = meta
        v = RegionResultView(meta, self._times, per_name)
        if where == "elem":
            self.elem_region[name] = v
        else:
            self.face_region[name] = v

    def __getitem__(self, key: str):
        """Look up a variable by name across all locations."""
        if key in self.node:
            return self.node[key]
        if key in self.node_region:
            return self.node_region[key]
        if key in self.elem_item:
            return self.elem_item[key]
        if key in self.face_item:
            return self.face_item[key]
        if key in self.elem_mult:
            return self.elem_mult[key]
        if key in self.face_mult:
            return self.face_mult[key]
        if key in self.elem_region:
            return self.elem_region[key]
        if key in self.face_region:
            return self.face_region[key]
        raise KeyError(key)

    def __repr__(self) -> str:
        return (
            f"Results(ntimes={len(self)}, "
            f"node={len(self.node)}, node_region={len(self.node_region)}, "
            f"elem_item={len(self.elem_item)}, elem_mult={len(self.elem_mult)}, elem_region={len(self.elem_region)}, "
            f"face_item={len(self.face_item)}, face_mult={len(self.face_mult)}, face_region={len(self.face_region)})"
        )

    __str__ = __repr__


# --------------------------- reader ---------------------------


class xplt:
    """FEBio ``.xplt`` reader.

    Features
    --------
    - Global and per-domain FMT_NODE.
    - FMT_ITEM, FMT_MULT, and FMT_REGION for domains and surfaces.
    - Builds a :class:`Mesh` with parts, surfaces, and nodesets.

    Typical use
    -----------
    >>> xp = xplt("run.xplt")
    >>> xp.readAllStates()
    >>> r = xp.results
    >>> disp = r["displacement"][0, ":", "x"]
    """

    def __init__(self, filename: str):
        self._reader = BinaryReader(filename)
        self._time: List[float] = []
        self._readMode = ""

        # dictionary
        self.dictionary: Dict[str, Dict[str, str]] = {}
        self._field_meta: Dict[str, _FieldMeta] = {}
        self._dict_order: Dict[str, List[str]] = {"node": [], "elem": [], "face": []}
        self._where: Dict[str, str] = {}  # var -> "node" | "elem" | "face"

        # ids and ordinals
        self._part_id2name: Dict[int, str] = {}
        self._surf_id2name: Dict[int, str] = {}
        self._dom_idx2name: Dict[int, str] = {}  # PLT_DOMAIN ordinal -> domain name
        self._surf_idx2name: Dict[int, str] = {}  # PLT_SURFACE ordinal -> surface name

        # mesh accumulators
        self._nodes_xyz: np.ndarray | None = None
        self._conn_list: List[np.ndarray] = []
        self._etype_list: List[str] = []
        self._parts_map: Dict[str, List[int]] = {}
        self._surfaces_map: Dict[str, List[np.ndarray]] = {}
        self._nodesets_map: Dict[str, np.ndarray] = {}

        # result buffers
        self._buf_node_global: Dict[str, List[np.ndarray | None]] = {}
        self._buf_node_region: Dict[str, Dict[str, List[np.ndarray | None]]] = {}
        self._buf_elem_item: Dict[str, Dict[str, List[np.ndarray | None]]] = {}
        self._buf_elem_mult: Dict[str, Dict[str, List[MultBlock | None]]] = {}
        self._buf_elem_region: Dict[str, Dict[str, List[np.ndarray | None]]] = {}
        self._buf_face_item: Dict[str, Dict[str, List[np.ndarray | None]]] = {}
        self._buf_face_mult: Dict[str, Dict[str, List[MultBlock | None]]] = {}
        self._buf_face_region: Dict[str, Dict[str, List[np.ndarray | None]]] = {}

        # per-state scratch
        self._state_node_global_seen: Dict[str, np.ndarray] = {}
        self._state_node_region_seen: Dict[str, Dict[str, np.ndarray]] = {}
        self._state_elem_item_seen: Dict[str, Dict[str, np.ndarray]] = {}
        self._state_elem_mult_seen: Dict[str, Dict[str, MultBlock]] = {}
        self._state_elem_region_seen: Dict[str, Dict[str, np.ndarray]] = {}
        self._state_face_item_seen: Dict[str, Dict[str, np.ndarray]] = {}
        self._state_face_mult_seen: Dict[str, Dict[str, MultBlock]] = {}
        self._state_face_region_seen: Dict[str, Dict[str, np.ndarray]] = {}

        # parse header+dict+mesh
        self._read_xplt()
        self.mesh = self._build_mesh()
        self._domain_nodes: Dict[str, np.ndarray] = self._compute_domain_nodes()

        self.version: int
        self.compression: int
        self.results: Results = Results([])

    def __repr__(self) -> str:
        return f"xplt(v={getattr(self, 'version', 'NA')}, comp={getattr(self, 'compression', 'NA')}, ntimes={len(self._time)}, vars={len(self.dictionary)})"

    __str__ = __repr__

    # mesh

    def _readMesh(self):
        """Parse node coordinates, domains, surfaces, and nodesets from the file."""
        self._reader.search_block("PLT_MESH")

        # nodes
        self._reader.search_block("PLT_NODE_SECTION")
        self._reader.search_block("PLT_NODE_HEADER")
        self._reader.search_block("PLT_NODE_SIZE")
        nodeSize = int(struct.unpack("I", self._reader.read(4))[0])
        self._reader.search_block("PLT_NODE_DIM")
        nodeDim = int(struct.unpack("I", self._reader.read(4))[0])
        self._reader.search_block("PLT_NODE_COORDS")
        xyz = np.zeros((nodeSize, max(3, nodeDim)), dtype=float)
        for _i in range(nodeSize):
            _ = struct.unpack("I", self._reader.read(4))[0]
            for j in range(nodeDim):
                xyz[_i, j] = struct.unpack("f", self._reader.read(4))[0]
        self._nodes_xyz = xyz

        # domains
        self._reader.search_block("PLT_DOMAIN_SECTION")
        dom_ord = 0
        while self._reader.check_block("PLT_DOMAIN"):
            self._reader.search_block("PLT_DOMAIN")
            self._reader.search_block("PLT_DOMAIN_HDR")
            self._reader.search_block("PLT_DOM_ELEM_TYPE")
            et_num = int(struct.unpack("I", self._reader.read(4))[0])

            self._reader.search_block("PLT_DOM_PART_ID")
            part_id_1b = int(struct.unpack("I", self._reader.read(4))[0])
            did = part_id_1b - 1

            self._reader.search_block("PLT_DOM_ELEMS")
            _ = int(struct.unpack("I", self._reader.read(4))[0])

            nlen = self._reader.search_block("PLT_DOM_NAME")
            dname = (
                self._reader.read(nlen)
                .split(b"\x00")[-1]
                .decode("utf-8", errors="ignore")
                if nlen > 0
                else None
            )
            part_name = dname or f"part_{did}"
            self._part_id2name[did] = part_name
            self._dom_idx2name[dom_ord] = part_name
            dom_ord += 1

            self._reader.search_block("PLT_DOM_ELEM_LIST")
            etype = Elem_Type(et_num).name
            nne = int(nodesPerElementClass[etype])

            while self._reader.check_block("PLT_ELEMENT"):
                sec_sz = self._reader.search_block("PLT_ELEMENT", print_tag=0)
                payload = self._reader.read(sec_sz)
                vals = np.frombuffer(payload, dtype=np.uint32, count=sec_sz // 4)
                if vals.size < nne:
                    raise ValueError(
                        f"PLT_ELEMENT too short for {etype} (need {nne}, got {vals.size})"
                    )
                nodes_raw = vals[-nne:]
                N = self._nodes_xyz.shape[0] if self._nodes_xyz is not None else 0
                nodes0 = _norm_node_ids(nodes_raw, N)
                self._conn_list.append(nodes0)
                self._etype_list.append(etype)
                self._parts_map.setdefault(part_name, []).append(
                    len(self._etype_list) - 1
                )

        # surfaces
        if self._reader.search_block("PLT_SURFACE_SECTION") > 0:
            surf_ord = 0
            while self._reader.check_block("PLT_SURFACE"):
                self._reader.search_block("PLT_SURFACE")
                self._reader.search_block("PLT_SURFACE_HDR")
                self._reader.search_block("PLT_SURFACE_ID")
                sid_1b = int(struct.unpack("I", self._reader.read(4))[0]) - 1
                self._reader.search_block("PLT_SURFACE_FACES")
                _ = int(struct.unpack("I", self._reader.read(4))[0])
                nlen = self._reader.seek_block("PLT_SURFACE_NAME")
                sname = (
                    self._reader.read(nlen)
                    .decode("utf-8", errors="ignore")
                    .split("\x00")[-1]
                )
                self._surf_id2name[sid_1b] = sname
                self._surf_idx2name[surf_ord] = sname
                surf_ord += 1

                self._reader.search_block("PLT_SURFACE_MAX_FACET_NODES")
                maxn = int(struct.unpack("I", self._reader.read(4))[0])

                if self._reader.check_block("PLT_FACE_LIST"):
                    self._reader.search_block("PLT_FACE_LIST")
                    lst: List[np.ndarray] = []
                    while self._reader.check_block("PLT_FACE"):
                        sec_size = self._reader.search_block("PLT_FACE")
                        cur = self._reader.tell()
                        _ = int(struct.unpack("I", self._reader.read(4))[0])
                        self._reader.skip(4)
                        face = np.zeros(maxn, dtype=np.int64)
                        for j in range(maxn):
                            face[j] = int(struct.unpack("I", self._reader.read(4))[0])
                        N = (
                            self._nodes_xyz.shape[0]
                            if self._nodes_xyz is not None
                            else 0
                        )
                        lst.append(_norm_node_ids(face, N))
                        self._reader.seek(cur + sec_size, SEEK_SET)
                    self._surfaces_map[sname] = lst

        # nodesets
        if self._reader.search_block("PLT_NODESET_SECTION") > 0:
            while self._reader.check_block("PLT_NODESET"):
                self._reader.search_block("PLT_NODESET")
                self._reader.search_block("PLT_NODESET_HDR")
                self._reader.search_block("PLT_NODESET_ID")
                _ = int(struct.unpack("I", self._reader.read(4))[0])
                self._reader.search_block("PLT_NODESET_SIZE")
                nsize = int(struct.unpack("I", self._reader.read(4))[0])
                nlen = self._reader.search_block("PLT_NODESET_NAME")
                nname = (
                    self._reader.read(nlen)
                    .decode("utf-8", errors="ignore")
                    .split("\x00")[-1]
                )
                ids: List[int] = []
                if self._reader.check_block("PLT_NODESET_LIST"):
                    self._reader.search_block("PLT_NODESET_LIST")
                    for _ in range(nsize):
                        ids.append(int(struct.unpack("I", self._reader.read(4))[0]))
                N = self._nodes_xyz.shape[0] if self._nodes_xyz is not None else 0
                self._nodesets_map[nname] = _norm_node_ids(
                    np.asarray(ids, dtype=np.int64), N
                )

        # optional part renames
        if self._reader.search_block("PLT_PARTS_SECTION") > 0:
            while self._reader.check_block("PLT_PART"):
                self._reader.search_block("PLT_PART")
                self._reader.search_block("PLT_PART_ID")
                pid = int(struct.unpack("I", self._reader.read(4))[0]) - 1
                nlen = self._reader.search_block("PLT_PART_NAME")
                pname = (
                    self._reader.read(nlen)
                    .decode("utf-8", errors="ignore")
                    .split("\x00")[0]
                )
                self._part_id2name[pid] = pname

    def _build_mesh(self) -> Mesh:
        """Build a :class:`Mesh` from parsed accumulators."""
        nodes = NodeArray(
            self._nodes_xyz if self._nodes_xyz is not None else np.zeros((0, 3), float)
        )
        kmax = max((len(c) for c in self._conn_list), default=0)
        E = len(self._conn_list)
        conn = -np.ones((E, kmax), dtype=np.int64)
        nper = np.zeros((E,), dtype=np.int64)
        for i, c in enumerate(self._conn_list):
            conn[i, : c.size] = c
            nper[i] = c.size
        elements = ElementArray(
            conn=conn, nper=nper, etype=np.asarray(self._etype_list, dtype=object)
        )
        parts = {
            name: np.asarray(idx, dtype=np.int64)
            for name, idx in self._parts_map.items()
        }
        surfaces: Dict[str, SurfaceArray] = {}
        for sname, lst in self._surfaces_map.items():
            if not lst:
                continue
            kk = max(len(a) for a in lst)
            F = len(lst)
            faces = -np.ones((F, kk), dtype=np.int64)
            nps = np.zeros((F,), dtype=np.int64)
            for i, a in enumerate(lst):
                faces[i, : a.size] = a
                nps[i] = a.size
            surfaces[sname] = SurfaceArray(faces=faces, nper=nps)
        nodesets = {name: ids for name, ids in self._nodesets_map.items()}
        return Mesh(
            nodes=nodes,
            elements=elements,
            parts=parts,
            surfaces=surfaces,
            nodesets=nodesets,
        )

    def _compute_domain_nodes(self) -> Dict[str, np.ndarray]:
        """Compute unique node lists per domain in original encounter order.

        The output maps each domain name to the list of global node ids that
        appear in its elements.
        """
        out: Dict[str, np.ndarray] = {}
        conn = self.mesh.elements.conn
        nper = self.mesh.elements.nper
        for dname, eids in self.mesh.parts.items():
            ids: List[int] = []
            for e in np.asarray(eids, dtype=np.int64):
                k = int(nper[e])
                if k > 0:
                    ids.extend(conn[e, :k].tolist())
            seen: set[int] = set()
            order: List[int] = []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    order.append(i)
            out[dname] = np.asarray(order, dtype=np.int64)
        return out

    # dictionary

    def _readDictStream(self, dictType: str, kind: str):
        """Read one dictionary stream for a given kind (node/elem/face)."""
        self._reader.search_block(dictType)
        while self._reader.check_block("PLT_DIC_ITEM"):
            self._reader.search_block("PLT_DIC_ITEM")
            self._reader.search_block("PLT_DIC_ITEM_TYPE")
            tnum = int(struct.unpack("I", self._reader.read(4))[0])
            self._reader.search_block("PLT_DIC_ITEM_FMT")
            fnum = int(struct.unpack("I", self._reader.read(4))[0])
            self._reader.search_block("PLT_DIC_ITEM_NAME")
            name = (
                self._reader.read(64).decode("utf-8", errors="ignore").split("\x00")[0]
            )
            self.dictionary[name] = {
                "type": FEDataType(tnum).name,
                "format": Storage_Fmt(fnum).name,
            }
            self._dict_order[kind].append(name)
            self._where[name] = kind

    def _readDict(self):
        """Read and initialize dictionary and result buffers."""
        self.dictionary.clear()
        self._dict_order = {"node": [], "elem": [], "face": []}
        self._where = {}

        self._reader.search_block("PLT_DICTIONARY")
        self._readDictStream("PLT_DIC_NODAL", "node")
        self._readDictStream("PLT_DIC_DOMAIN", "elem")
        self._readDictStream("PLT_DIC_SURFACE", "face")

        self._field_meta.clear()
        self._buf_node_global.clear()
        self._buf_node_region.clear()
        self._buf_elem_item.clear()
        self._buf_elem_mult.clear()
        self._buf_elem_region.clear()
        self._buf_face_item.clear()
        self._buf_face_mult.clear()
        self._buf_face_region.clear()

        for name, d in self.dictionary.items():
            dtype = FEDataType[d["type"]]
            fmt = Storage_Fmt[d["format"]]
            fm = _FieldMeta(name, fmt, dtype)
            self._field_meta[name] = fm
            kind = self._where[name]
            if kind == "node" and fmt == Storage_Fmt.FMT_NODE:
                self._buf_node_global[name] = []
                self._buf_node_region[name] = {}
            elif kind == "elem":
                if fmt == Storage_Fmt.FMT_ITEM:
                    self._buf_elem_item[name] = {}
                elif fmt == Storage_Fmt.FMT_MULT:
                    self._buf_elem_mult[name] = {}
                elif fmt == Storage_Fmt.FMT_REGION:
                    self._buf_elem_region[name] = {}
            elif kind == "face":
                if fmt == Storage_Fmt.FMT_ITEM:
                    self._buf_face_item[name] = {}
                elif fmt == Storage_Fmt.FMT_MULT:
                    self._buf_face_mult[name] = {}
                elif fmt == Storage_Fmt.FMT_REGION:
                    self._buf_face_region[name] = {}

    # states

    @staticmethod
    def _pack_mult(block_flat: np.ndarray, nper: np.ndarray, C: int) -> np.ndarray:
        """Pack a flat MULT block into a padded 3D array."""
        R = int(nper.size)
        Kmax = int(nper.max()) if R > 0 else 0
        out = np.full((R, Kmax, C), np.nan, dtype=np.float32)
        off = 0
        for r in range(R):
            k = int(nper[r])
            if k > 0:
                out[r, :k, :] = block_flat[off : off + k, :]
                off += k
        return out

    def _flush_state_missing(self):
        """Append per-state buffers and mark missing entries as None."""
        # node global
        for v in self._buf_node_global.keys():
            self._buf_node_global[v].append(self._state_node_global_seen.get(v, None))
        # node per-region
        dom_names = list(self._dom_idx2name.values())
        for v in list(self._buf_node_region.keys()):
            if not self._buf_node_region[v]:
                for d in dom_names:
                    self._buf_node_region[v][d] = []
            seen = self._state_node_region_seen.get(v, {})
            for d in self._buf_node_region[v].keys():
                self._buf_node_region[v][d].append(seen.get(d, None))

        # elem item/mult/region
        for v in list(self._buf_elem_item.keys()):
            if not self._buf_elem_item[v]:
                for d in dom_names:
                    self._buf_elem_item[v][d] = []
            seen = self._state_elem_item_seen.get(v, {})
            for d in self._buf_elem_item[v].keys():
                self._buf_elem_item[v][d].append(seen.get(d, None))

        for v in list(self._buf_elem_mult.keys()):
            if not self._buf_elem_mult[v]:
                for d in dom_names:
                    self._buf_elem_mult[v][d] = []
            seen = self._state_elem_mult_seen.get(v, {})
            for d in self._buf_elem_mult[v].keys():
                self._buf_elem_mult[v][d].append(seen.get(d, None))

        for v in list(self._buf_elem_region.keys()):
            if not self._buf_elem_region[v]:
                for d in dom_names:
                    self._buf_elem_region[v][d] = []
            seen = self._state_elem_region_seen.get(v, {})
            for d in self._buf_elem_region[v].keys():
                self._buf_elem_region[v][d].append(seen.get(d, None))

        # face item/mult/region
        surf_names = list(self._surf_idx2name.values())
        for v in list(self._buf_face_item.keys()):
            if not self._buf_face_item[v]:
                for s in surf_names:
                    self._buf_face_item[v][s] = []
            seen = self._state_face_item_seen.get(v, {})
            for s in self._buf_face_item[v].keys():
                self._buf_face_item[v][s].append(seen.get(s, None))

        for v in list(self._buf_face_mult.keys()):
            if not self._buf_face_mult[v]:
                for s in surf_names:
                    self._buf_face_mult[v][s] = []
            seen = self._state_face_mult_seen.get(v, {})
            for s in self._buf_face_mult[v].keys():
                self._buf_face_mult[v][s].append(seen.get(s, None))

        for v in list(self._buf_face_region.keys()):
            if not self._buf_face_region[v]:
                for s in surf_names:
                    self._buf_face_region[v][s] = []
            seen = self._state_face_region_seen.get(v, {})
            for s in self._buf_face_region[v].keys():
                self._buf_face_region[v][s].append(seen.get(s, None))

        # clear scratch
        self._state_node_global_seen.clear()
        self._state_node_region_seen.clear()
        self._state_elem_item_seen.clear()
        self._state_elem_mult_seen.clear()
        self._state_elem_region_seen.clear()
        self._state_face_item_seen.clear()
        self._state_face_mult_seen.clear()
        self._state_face_region_seen.clear()

    def _readResultStream(self, tag: str, kind: str):
        """Read a result stream for node/elem/face in the current state."""
        order = self._dict_order[kind]
        _ = self._reader.search_block(tag)
        while self._reader.check_block("PLT_STATE_VARIABLE"):
            self._reader.search_block("PLT_STATE_VARIABLE")
            self._reader.search_block("PLT_STATE_VAR_ID")
            var_id_1b = int(struct.unpack("I", self._reader.read(4))[0])
            idx = var_id_1b - 1  # 1-based -> 0-based

            dlen = self._reader.search_block("PLT_STATE_VAR_DATA")
            endp = self._reader.tell() + dlen

            if idx < 0 or idx >= len(order):
                self._reader.seek(endp, SEEK_SET)
                continue

            dictKey = order[idx]
            meta = self._field_meta.get(dictKey)
            if meta is None:
                self._reader.seek(endp, SEEK_SET)
                continue
            C = meta.ncomp

            while self._reader.tell() < endp:
                reg_raw = int(struct.unpack("I", self._reader.read(4))[0])
                size_b = int(struct.unpack("I", self._reader.read(4))[0])
                nrows = int(size_b // (C * 4))
                if nrows <= 0:
                    continue

                # NODAL (global or per-domain)
                if kind == "node" and meta.fmt == Storage_Fmt.FMT_NODE:
                    block = np.frombuffer(
                        self._reader.read(4 * C * nrows), dtype=np.float32
                    ).reshape(nrows, C)
                    rid = reg_raw - 1
                    rname = self._dom_idx2name.get(rid)
                    if rname is not None:
                        self._state_node_region_seen.setdefault(dictKey, {})[rname] = (
                            block
                        )
                    else:
                        self._state_node_global_seen[dictKey] = block
                    continue

                # resolve region name + nper for elem/face
                if kind == "elem":
                    rid = reg_raw - 1
                    rname = self._dom_idx2name.get(rid, f"domain_{rid}")
                    elem_ids = np.asarray(
                        self._parts_map.get(rname, []), dtype=np.int64
                    )
                    nper = (
                        self.mesh.elements.nper[elem_ids]
                        if elem_ids.size > 0
                        else np.zeros((0,), dtype=np.int64)
                    )
                else:
                    rid = reg_raw - 1
                    rname = self._surf_idx2name.get(
                        rid, self._surf_id2name.get(rid, f"surface_{rid}")
                    )
                    nper = (
                        self.mesh.surfaces[rname].nper
                        if rname in self.mesh.surfaces
                        else np.zeros((0,), dtype=np.int64)
                    )

                # ITEM
                if meta.fmt == Storage_Fmt.FMT_ITEM:
                    block = np.frombuffer(
                        self._reader.read(4 * C * nrows), dtype=np.float32
                    ).reshape(nrows, C)
                    if kind == "elem":
                        self._state_elem_item_seen.setdefault(dictKey, {})[rname] = (
                            block
                        )
                    else:
                        self._state_face_item_seen.setdefault(dictKey, {})[rname] = (
                            block
                        )
                    continue

                # MULT
                if meta.fmt == Storage_Fmt.FMT_MULT:
                    flat = np.frombuffer(
                        self._reader.read(4 * C * nrows), dtype=np.float32
                    ).reshape(nrows, C)
                    packed = self._pack_mult(flat, nper.astype(np.int64), C)
                    mb = MultBlock(
                        packed.astype(np.float32, copy=False),
                        nper.astype(np.int64, copy=False),
                    )
                    if kind == "elem":
                        self._state_elem_mult_seen.setdefault(dictKey, {})[rname] = mb
                    else:
                        self._state_face_mult_seen.setdefault(dictKey, {})[rname] = mb
                    continue

                # REGION
                if meta.fmt == Storage_Fmt.FMT_REGION:
                    vec = np.frombuffer(
                        self._reader.read(4 * C * nrows), dtype=np.float32
                    ).reshape(nrows, C)
                    row = vec[0, :] if vec.ndim == 2 else vec.reshape(C)
                    if kind == "elem":
                        self._state_elem_region_seen.setdefault(dictKey, {})[rname] = (
                            row
                        )
                    else:
                        self._state_face_region_seen.setdefault(dictKey, {})[rname] = (
                            row
                        )
                    continue

                # skip mismatched
                self._reader.seek(self._reader.tell() + 4 * C * nrows, SEEK_SET)

    def _readState(self) -> int:
        """Read one PLT_STATE. Returns 0 on success and non-zero to stop."""
        self._reader.search_block("PLT_STATE")
        self._reader.search_block("PLT_STATE_HEADER")
        self._reader.search_block("PLT_STATE_HDR_TIME")
        t = float(struct.unpack("f", self._reader.read(4))[0])
        self._reader.search_block("PLT_STATE_STATUS")
        status = int(struct.unpack("I", self._reader.read(4))[0])
        if status != 0:
            return 1
        self._time.append(t)
        self._reader.search_block("PLT_STATE_DATA")
        try:
            self._readResultStream("PLT_NODE_DATA", "node")
            self._readResultStream("PLT_ELEMENT_DATA", "elem")
            self._readResultStream("PLT_FACE_DATA", "face")
        finally:
            self._flush_state_missing()
        return 0

    def _skipState(self):
        """Skip one PLT_STATE block."""
        size = self._reader.search_block("PLT_STATE")
        if size < 0:
            raise RuntimeError("No further PLT_STATE found while skipping")
        self._reader.skip(size)

    def readAllStates(self):
        """Read all states in order and finalize results."""
        if self._readMode == "readSteps":
            raise RuntimeError("readAllStates incompatible with readSteps")
        while True:
            try:
                if self._readState() != 0:
                    break
            except Exception:
                break
        self._readMode = "readAllStates"
        self._finalize_results()

    def readSteps(self, stepList: List[int]):
        """Read a selected list of step indices and finalize results.

        Steps are 0-based. The method advances by skipping blocks between
        requested steps.
        """
        if self._readMode == "readAllStates":
            raise RuntimeError("readSteps incompatible with readAllStates")
        for i, s in enumerate(stepList):
            stepDiff = s - (stepList[i - 1] if i > 0 else 0) - (1 if i > 0 else 0)
            for _ in range(stepDiff):
                self._skipState()
            self._readState()
        self._readMode = "readSteps"
        self._finalize_results()
        self._reader.close()

    # finalize

    def _finalize_results(self):
        """Convert buffers into views and attach them to the Results object."""
        times = np.asarray(self._time, float)
        self.results = Results(times)

        # node global
        for name, per_t in self._buf_node_global.items():
            if any(a is not None for a in per_t):
                sizes = {a.shape[0] for a in per_t if a is not None}
                if len(sizes) > 1:
                    raise ValueError(f"{name}: inconsistent nodal row count over time")
                self.results.register_node_global(
                    name, self._field_meta[name], per_t, self.mesh
                )

        # node region
        for name, per_name in self._buf_node_region.items():
            has_any = any(any(a is not None for a in lst) for lst in per_name.values())
            if has_any:
                self.results.register_node_region(
                    name, self._field_meta[name], per_name, self._domain_nodes
                )

        # elem
        for name, per_name in self._buf_elem_item.items():
            self.results.register_item("elem", name, self._field_meta[name], per_name)
        for name, per_name in self._buf_elem_mult.items():
            self.results.register_mult("elem", name, self._field_meta[name], per_name)
        for name, per_name in self._buf_elem_region.items():
            self.results.register_region("elem", name, self._field_meta[name], per_name)

        # face
        for name, per_name in self._buf_face_item.items():
            self.results.register_item("face", name, self._field_meta[name], per_name)
        for name, per_name in self._buf_face_mult.items():
            self.results.register_mult("face", name, self._field_meta[name], per_name)
        for name, per_name in self._buf_face_region.items():
            self.results.register_region("face", name, self._field_meta[name], per_name)

        # clear
        self._buf_node_global.clear()
        self._buf_node_region.clear()
        self._buf_elem_item.clear()
        self._buf_elem_mult.clear()
        self._buf_elem_region.clear()
        self._buf_face_item.clear()
        self._buf_face_mult.clear()
        self._buf_face_region.clear()

    # header

    def _read_xplt(self):
        """Read file header then dictionary and mesh sections."""
        magic = int(struct.unpack("I", self._reader.read(4))[0])
        if magic != 4605250:
            raise RuntimeError("Not a valid xplt")
        self._reader.search_block("PLT_ROOT")
        self._reader.search_block("PLT_HEADER")
        self._reader.search_block("PLT_HDR_VERSION")
        self.version = int(struct.unpack("I", self._reader.read(4))[0])
        self._reader.search_block("PLT_HDR_COMPRESSION")
        self.compression = int(struct.unpack("I", self._reader.read(4))[0])
        self._readDict()
        self._readMesh()
