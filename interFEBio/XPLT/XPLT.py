# XPLT.py — FEBio .xplt reader wired to your standalone Mesh class.
# Mesh API: interFEBio.Mesh.Mesh
# Enums:    .Enums
# Reader:   .BinaryReader.BinaryReader

from __future__ import annotations

import struct
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

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _norm_node_ids(ids: np.ndarray, N: int) -> np.ndarray:
    """Normalize node ids to zero-based. Accepts padded negatives."""
    a = np.asarray(ids, dtype=np.int64, order="C")
    if a.size == 0:
        return a
    valid = a >= 0
    if not np.any(valid):
        return a
    vmax = int(a[valid].max())
    # 1-based if highest referenced id equals N -> shift down
    if vmax == N:
        a[valid] -= 1
    elif vmax > N:
        raise ValueError(f"node id {vmax} exceeds node count {N}")
    # else already zero-based
    if np.any(a[valid] < 0) or np.any(a[valid] >= N):
        raise ValueError("normalized node ids out of range")
    return a


# ---------------------------------------------------------------------
# Field meta
# ---------------------------------------------------------------------


class _FieldMeta:
    __slots__ = ("name", "fmt", "dtype", "ncomp")

    def __init__(self, name: str, fmt: Storage_Fmt, dtype: FEDataType):
        self.name = name
        self.fmt = fmt
        self.dtype = dtype
        self.ncomp = int(FEDataDim[dtype.name].value)

    def __repr__(self) -> str:  # pragma: no cover
        return f"_FieldMeta(name={self.name!r}, fmt={self.fmt.name}, dtype={self.dtype.name}, ncomp={self.ncomp})"

    __str__ = __repr__


# ---------------------------------------------------------------------
# Component string mapping
# ---------------------------------------------------------------------

Index = int | slice | Iterable[int] | np.ndarray | None

_VEC3_ORDER = ("x", "y", "z")
_MAT3FD_ORDER = ("xx", "yy", "zz")
_MAT3FS_ORDER = ("xx", "yy", "zz", "xy", "yz", "xz")  # symmetric Voigt
_MAT3F_ORDER = ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")  # row-major 3x3
_VOIGT6 = {"xx": 0, "yy": 1, "zz": 2, "yz": 3, "xz": 4, "xy": 5}


def _normalize_comp_token(s: str) -> str:
    return s.strip().lower().replace(" ", "")


def _comp_names_for_dtype(dtype: FEDataType) -> tuple[str, ...] | None:
    if dtype == FEDataType.VEC3F:
        return _VEC3_ORDER
    if dtype == FEDataType.MAT3FD:
        return _MAT3FD_ORDER
    if dtype == FEDataType.MAT3FS:
        return _MAT3FS_ORDER
    if dtype == FEDataType.MAT3F:
        return _MAT3F_ORDER
    return None  # TENS4FS handled separately


def _tens4fs_pair_index(p: str) -> int:
    p = _normalize_comp_token(p)
    if len(p) != 4:
        raise KeyError(f"invalid 4th-order token '{p}', use 'xxyy','xyxy','yzyz'")
    a, b = p[:2], p[2:]
    if a not in _VOIGT6 or b not in _VOIGT6:
        raise KeyError(f"invalid pair in '{p}'")
    i, j = _VOIGT6[a], _VOIGT6[b]
    if i > j:
        i, j = j, i
    return i + (j * (j + 1)) // 2  # 0..20


def _comp_spec_to_index(meta: _FieldMeta, spec: Index | str | Iterable[str]) -> Index:
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
            except ValueError as error:
                raise KeyError(
                    f"component '{spec}' not valid for {meta.dtype.name}. Valid: {names}"
                ) from error
        if meta.dtype == FEDataType.TENS4FS:
            return _tens4fs_pair_index(s)
        raise KeyError(f"component strings not supported for {meta.dtype.name}")
    # list/tuple of strings
    try:
        lst = list(spec)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError("unsupported component selector") from error
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


# ---------------------------------------------------------------------
# Results views with NaN-filling for missing states
# ---------------------------------------------------------------------


def _as_index_list(sel: Index, T: int) -> List[int]:
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
    if s is None:
        return n
    if isinstance(s, int):
        return 1
    if isinstance(s, slice):
        start, stop, step = s.indices(n)
        return max(0, (stop - start + (step - 1)) // step)
    return len(list(s))  # type: ignore[arg-type]


class _BaseView:
    __slots__ = ("meta", "_times", "_comp_idx", "_t_idx")

    def __init__(self, meta: _FieldMeta, times: np.ndarray):
        self.meta = meta
        self._times = np.asarray(times, float)
        self._comp_idx: Index = None
        self._t_idx: Index = None

    def time(self, idx: Index | str = None):
        if idx == ":":
            idx = slice(None)
        self._t_idx = idx  # type: ignore[assignment]
        return self

    def comp(self, idx: Index | str | Iterable[str]):
        self._comp_idx = _comp_spec_to_index(self.meta, idx)
        return self.eval()

    def dims(self) -> tuple[str, ...]:
        raise NotImplementedError

    def eval(self) -> np.ndarray:
        raise NotImplementedError

    def __len__(self) -> int:
        return int(self._times.shape[0])


class NodeResultView(_BaseView):
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
        self._node_idx = ids
        return self

    def nodeset(self, name: str):
        self._node_idx = self._mesh.nodesets[name]
        return self

    def dims(self) -> tuple[str, ...]:
        return ("time", "node", "component")

    def eval(self) -> np.ndarray:
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
        if not isinstance(key, tuple):
            t = key
            self._t_idx = t if t != ":" else slice(None)  # type: ignore[assignment]
            self._node_idx = None
            self._comp_idx = None
            return self.eval()
        t, n, c = (key + (slice(None),) * 3)[:3]
        if t == ":":
            t = slice(None)
        if n == ":":
            n = slice(None)
        if isinstance(c, str):
            c = _comp_spec_to_index(self.meta, c)
        self._t_idx = t  # type: ignore[assignment]
        self._node_idx = n  # type: ignore[assignment]
        self._comp_idx = c  # type: ignore[assignment]
        return self.eval()

    def __repr__(self) -> str:  # pragma: no cover
        N = next((a.shape[0] for a in self._per_t if a is not None), 0)
        C = next((a.shape[1] for a in self._per_t if a is not None), self.meta.ncomp)
        missing = sum(1 for a in self._per_t if a is None)
        return f"NodeResultView(name={self.meta.name!r}, T={len(self)}, N={N}, C={C}, missing={missing})"

    __str__ = __repr__


class ElemResultView(_BaseView):
    __slots__ = ("_per_name", "_elem_idx")

    def __init__(
        self,
        meta: _FieldMeta,
        times: np.ndarray,
        per_name: Dict[str, List[np.ndarray | None]],
    ):
        super().__init__(meta, times)
        self._per_name = per_name
        self._elem_idx: Index = None

    def domains(self) -> List[str]:
        return sorted(self._per_name.keys())

    def domain(self, name: str):
        if name not in self._per_name:
            raise KeyError(name)
        return ElemResultView(self.meta, self._times, {name: self._per_name[name]})

    def elems(self, ids: Index):
        self._elem_idx = ids
        return self

    def dims(self) -> tuple[str, ...]:
        return ("time", "element", "component")

    def _pick_per_t(self) -> List[np.ndarray | None]:
        if len(self._per_name) != 1:
            raise ValueError("multiple domains present; select a domain() first")
        return next(iter(self._per_name.values()))

    def eval(self, *, domain: str | None = None) -> np.ndarray:
        per_t = self._pick_per_t() if domain is None else self._per_name[domain]
        T_sel = _as_index_list(self._t_idx, len(self))
        R0 = next((a.shape[0] for a in per_t if a is not None), 0)
        C0 = self.meta.ncomp
        R_sel = _shape_from_slice(R0, self._elem_idx)
        C_sel = _shape_from_slice(C0, self._comp_idx)

        def _one(k: int) -> np.ndarray:
            a = per_t[k]
            if a is None:
                return np.full((R_sel, C_sel), np.nan, dtype=np.float32)
            out = a
            if self._elem_idx is not None:
                out = out[self._elem_idx, :]  # type: ignore[index]
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
        _ = self._pick_per_t()
        if not isinstance(key, tuple):
            t = key
            self._t_idx = t if t != ":" else slice(None)  # type: ignore[assignment]
            self._elem_idx = None
            self._comp_idx = None
            return self.eval()
        t, e, c = (key + (slice(None),) * 3)[:3]
        if t == ":":
            t = slice(None)
        if e == ":":
            e = slice(None)
        if isinstance(c, str):
            c = _comp_spec_to_index(self.meta, c)
        self._t_idx = t  # type: ignore[assignment]
        self._elem_idx = e  # type: ignore[assignment]
        self._comp_idx = c  # type: ignore[assignment]
        return self.eval()

    def __repr__(self) -> str:  # pragma: no cover
        names = list(self._per_name.keys())
        return f"ElemResultView(name={self.meta.name!r}, domains={names})"

    __str__ = __repr__


class FaceResultView(_BaseView):
    __slots__ = ("_per_name", "_face_idx")

    def __init__(
        self,
        meta: _FieldMeta,
        times: np.ndarray,
        per_name: Dict[str, List[np.ndarray | None]],
    ):
        super().__init__(meta, times)
        self._per_name = per_name
        self._face_idx: Index = None

    def surfaces(self) -> List[str]:
        return sorted(self._per_name.keys())

    def surface(self, name: str):
        if name not in self._per_name:
            raise KeyError(name)
        return FaceResultView(self.meta, self._times, {name: self._per_name[name]})

    def faces(self, ids: Index):
        self._face_idx = ids
        return self

    def dims(self) -> tuple[str, ...]:
        return ("time", "face", "component")

    def _pick_per_t(self) -> List[np.ndarray | None]:
        if len(self._per_name) != 1:
            raise ValueError("multiple surfaces present; select a surface() first")
        return next(iter(self._per_name.values()))

    def eval(self, *, surface: str | None = None) -> np.ndarray:
        per_t = self._pick_per_t() if surface is None else self._per_name[surface]
        T_sel = _as_index_list(self._t_idx, len(self))
        R0 = next((a.shape[0] for a in per_t if a is not None), 0)
        C0 = self.meta.ncomp
        R_sel = _shape_from_slice(R0, self._face_idx)
        C_sel = _shape_from_slice(C0, self._comp_idx)

        def _one(k: int) -> np.ndarray:
            a = per_t[k]
            if a is None:
                return np.full((R_sel, C_sel), np.nan, dtype=np.float32)
            out = a
            if self._face_idx is not None:
                out = out[self._face_idx, :]  # type: ignore[index]
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
        _ = self._pick_per_t()
        if not isinstance(key, tuple):
            t = key
            self._t_idx = t if t != ":" else slice(None)  # type: ignore[assignment]
            self._face_idx = None
            self._comp_idx = None
            return self.eval()
        t, f, c = (key + (slice(None),) * 3)[:3]
        if t == ":":
            t = slice(None)
        if f == ":":
            f = slice(None)
        if isinstance(c, str):
            c = _comp_spec_to_index(self.meta, c)
        self._t_idx = t  # type: ignore[assignment]
        self._face_idx = f  # type: ignore[assignment]
        self._comp_idx = c  # type: ignore[assignment]
        return self.eval()

    def __repr__(self) -> str:  # pragma: no cover
        names = list(self._per_name.keys())
        return f"FaceResultView(name={self.meta.name!r}, surfaces={names})"

    __str__ = __repr__


class Results:
    def __init__(self, times: Iterable[float]):
        self._times = np.asarray(list(times), float)
        self._nodal: Dict[str, NodeResultView] = {}
        self._elem: Dict[str, ElemResultView] = {}
        self._face: Dict[str, FaceResultView] = {}
        self._meta: Dict[str, _FieldMeta] = {}

    def __len__(self) -> int:
        return int(self._times.shape[0])

    def register_nodal(
        self, name: str, meta: _FieldMeta, per_t: List[np.ndarray | None], mesh: Mesh
    ):
        self._meta[name] = meta
        self._nodal[name] = NodeResultView(meta, self._times, per_t, mesh)

    def register_elem(
        self, name: str, meta: _FieldMeta, per_name: Dict[str, List[np.ndarray | None]]
    ):
        self._meta[name] = meta
        self._elem[name] = ElemResultView(meta, self._times, per_name)

    def register_face(
        self, name: str, meta: _FieldMeta, per_name: Dict[str, List[np.ndarray | None]]
    ):
        self._meta[name] = meta
        self._face[name] = FaceResultView(meta, self._times, per_name)

    def times(self) -> np.ndarray:
        return self._times

    def __getitem__(self, key: str):
        if key in self._nodal:
            return self._nodal[key]
        if key in self._elem:
            return self._elem[key]
        if key in self._face:
            return self._face[key]
        raise KeyError(key)

    def keys(self) -> List[str]:
        return list(self._meta.keys())

    def __repr__(self) -> str:  # pragma: no cover
        return f"Results(ntimes={len(self)}, vars={len(self._meta)})"

    __str__ = __repr__


# ---------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------


class xplt:
    """Reader for FEBio .xplt with mesh isolation and NaN-filled sparse results."""

    def __init__(self, filename: str):
        self._reader = BinaryReader(filename)
        self._time: List[float] = []
        self._readMode = ""

        # dictionary + meta
        self.dictionary: Dict[str, Dict[str, str]] = {}
        self._field_meta: Dict[str, _FieldMeta] = {}

        # id maps
        self._part_id2name: Dict[int, str] = {}
        self._surf_id2name: Dict[int, str] = {}

        # mesh accumulators
        self._nodes_xyz: np.ndarray | None = None
        self._conn_list: List[np.ndarray] = []
        self._etype_list: List[str] = []
        self._parts_map: Dict[str, List[int]] = {}
        self._surfaces_map: Dict[str, List[np.ndarray]] = {}
        self._nodesets_map: Dict[str, np.ndarray] = {}

        # results accumulators with per-time alignment
        self._buf_node: Dict[str, List[np.ndarray | None]] = {}
        self._buf_elem: Dict[str, Dict[str, List[np.ndarray | None]]] = {}
        self._buf_face: Dict[str, Dict[str, List[np.ndarray | None]]] = {}

        # per-state scratch
        self._state_node_seen: Dict[str, np.ndarray] = {}
        self._state_elem_seen: Dict[str, Dict[str, np.ndarray]] = {}
        self._state_face_seen: Dict[str, Dict[str, np.ndarray]] = {}

        self._read_xplt()  # header + dictionary + mesh
        self.mesh = self._build_mesh()

        self.version: int
        self.compression: int
        self.results: Results = Results([])

    def __repr__(self) -> str:  # pragma: no cover
        return f"xplt(v={getattr(self, 'version', 'NA')}, comp={getattr(self, 'compression', 'NA')}, ntimes={len(self._time)}, vars={len(self.dictionary)})"

    __str__ = __repr__

    # ------------------ mesh parsing -> Mesh ------------------

    def _readMesh(self):
        # Nodes
        self._reader.search_block("PLT_MESH")
        self._reader.search_block("PLT_NODE_SECTION")
        self._reader.search_block("PLT_NODE_HEADER")
        self._reader.search_block("PLT_NODE_SIZE")
        nodeSize = int(struct.unpack("I", self._reader.read(4))[0])
        self._reader.search_block("PLT_NODE_DIM")
        nodeDim = int(struct.unpack("I", self._reader.read(4))[0])
        self._reader.search_block("PLT_NODE_COORDS")
        xyz = np.zeros((nodeSize, max(3, nodeDim)), dtype=float)
        for i in range(nodeSize):
            _ = struct.unpack("I", self._reader.read(4))[0]  # node id
            for j in range(nodeDim):
                xyz[i, j] = struct.unpack("f", self._reader.read(4))[0]
        self._nodes_xyz = xyz

        # Domains
        self._reader.search_block("PLT_DOMAIN_SECTION")
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

        # Surfaces
        if self._reader.search_block("PLT_SURFACE_SECTION") > 0:
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
                self._reader.search_block("PLT_SURFACE_MAX_FACET_NODES")
                maxn = int(struct.unpack("I", self._reader.read(4))[0])

                if self._reader.check_block("PLT_FACE_LIST"):
                    self._reader.search_block("PLT_FACE_LIST")
                    lst: List[np.ndarray] = []
                    while self._reader.check_block("PLT_FACE"):
                        sec_size = self._reader.search_block("PLT_FACE")
                        cur = self._reader.tell()
                        _ = int(struct.unpack("I", self._reader.read(4))[0])  # face id
                        self._reader.skip(4)  # skip type
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

        # Nodesets
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

        # Parts rename (optional)
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

    # ------------------ dictionary ------------------

    def _readDictStream(self, dictType: str):
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

    def _readDict(self):
        self.dictionary.clear()
        self._reader.search_block("PLT_DICTIONARY")
        self._readDictStream("PLT_DIC_NODAL")
        self._readDictStream("PLT_DIC_DOMAIN")
        self._readDictStream("PLT_DIC_SURFACE")

        self._field_meta.clear()
        self._buf_node.clear()
        self._buf_elem.clear()
        self._buf_face.clear()

        for name, d in self.dictionary.items():
            dtype = FEDataType[d["type"]]
            fmt = Storage_Fmt[d["format"]]
            fm = _FieldMeta(name, fmt, dtype)
            self._field_meta[name] = fm
            if fmt == Storage_Fmt.FMT_NODE:
                self._buf_node[name] = []
            elif fmt == Storage_Fmt.FMT_ITEM:
                self._buf_elem[name] = {}
            else:
                self._buf_face[name] = {}

    # ------------------ states ------------------

    def _flush_state_missing(self):
        for v in self._buf_node.keys():
            self._buf_node[v].append(self._state_node_seen.get(v, None))
        for v in self._buf_elem.keys():
            if not self._buf_elem[v]:
                for pname in self._parts_map.keys():
                    self._buf_elem[v][pname] = []
            seen = self._state_elem_seen.get(v, {})
            for pname in self._buf_elem[v].keys():
                self._buf_elem[v][pname].append(seen.get(pname, None))
        for v in self._buf_face.keys():
            if not self._buf_face[v]:
                for sname in self._surfaces_map.keys():
                    self._buf_face[v][sname] = []
            seen = self._state_face_seen.get(v, {})
            for sname in self._buf_face[v].keys():
                self._buf_face[v][sname].append(seen.get(sname, None))
        self._state_node_seen.clear()
        self._state_elem_seen.clear()
        self._state_face_seen.clear()

    def _readResultStream(self, tag: str, kind: str):
        _ = self._reader.search_block(tag)
        while self._reader.check_block("PLT_STATE_VARIABLE"):
            self._reader.search_block("PLT_STATE_VARIABLE")
            self._reader.search_block("PLT_STATE_VAR_ID")
            _ = self._reader.read(4)  # unused
            dlen = self._reader.search_block("PLT_STATE_VAR_DATA")
            endp = self._reader.tell() + dlen

            dictKey = list(self._field_meta.keys())[self._var]
            C = self._field_meta[dictKey].ncomp

            while self._reader.tell() < endp:
                dom_raw = int(struct.unpack("I", self._reader.read(4))[0])
                size_b = int(struct.unpack("I", self._reader.read(4))[0])
                nrows = int(size_b // (C * 4))
                if nrows <= 0:
                    continue
                block = np.frombuffer(
                    self._reader.read(4 * C * nrows), dtype=np.float32
                ).reshape(nrows, C)

                if kind == "node":
                    self._state_node_seen[dictKey] = block
                elif kind == "elem":
                    did = dom_raw - 1
                    pname = self._part_id2name.get(did, f"part_{did}")
                    self._state_elem_seen.setdefault(dictKey, {})[pname] = block
                else:
                    sid = dom_raw - 1
                    sname = self._surf_id2name.get(sid, f"surface_{sid}")
                    self._state_face_seen.setdefault(dictKey, {})[sname] = block

            self._var += 1

    def _readState(self) -> int:
        self._var = 0
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
        size = self._reader.search_block("PLT_STATE")
        if size < 0:
            raise RuntimeError("No further PLT_STATE found while skipping")
        self._reader.skip(size)

    def readAllStates(self):
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

    # ------------------ finalize ------------------

    def _finalize_results(self):
        times = np.asarray(self._time, float)
        self.results = Results(times)

        for name, per_t in self._buf_node.items():
            sizes = {a.shape[0] for a in per_t if a is not None}
            if len(sizes) > 1:
                raise ValueError(f"{name}: inconsistent nodal row count over time")
            self.results.register_nodal(name, self._field_meta[name], per_t, self.mesh)

        for name, per_name in self._buf_elem.items():
            self.results.register_elem(name, self._field_meta[name], per_name)

        for name, per_name in self._buf_face.items():
            self.results.register_face(name, self._field_meta[name], per_name)

        self._buf_node.clear()
        self._buf_elem.clear()
        self._buf_face.clear()

    # ------------------ header ------------------

    def _read_xplt(self):
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
