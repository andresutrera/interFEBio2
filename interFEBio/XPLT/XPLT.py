from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

# Project enums
from .Enums import FEDataType, FEDataDim, Storage_Fmt, Elem_Type, tags
# New mesh classes
from interFEBio.Mesh.Mesh import Mesh, NodeArray, ElementArray, SurfaceArray


# ---------------------------------------------------------------------
# Fast, safe binary block reader
# ---------------------------------------------------------------------

class _Bin:
    """Minimal FEBio XPLT reader with safe block navigation."""

    __slots__ = ("f", "size")

    def __init__(self, path: str) -> None:
        self.f = open(path, "rb", buffering=1024 * 1024)
        self.f.seek(0, os.SEEK_END)
        self.size = self.f.tell()
        self.f.seek(0)

    def close(self) -> None:
        self.f.close()

    def tell(self) -> int:
        return self.f.tell()

    def seek(self, off: int, whence: int = os.SEEK_SET) -> None:
        self.f.seek(off, whence)

    def read_u32(self) -> int:
        buf = self.f.read(4)
        if len(buf) < 4:
            raise EOFError("u32 read past EOF")
        return struct.unpack("<I", buf)[0]

    def read_f32(self) -> float:
        buf = self.f.read(4)
        if len(buf) < 4:
            raise EOFError("f32 read past EOF")
        return struct.unpack("<f", buf)[0]

    def read_bytes(self, n: int) -> bytes:
        return self.f.read(n)

    def _peek_tag(self) -> int:
        pos = self.f.tell()
        buf = self.f.read(4)
        self.f.seek(pos)
        if len(buf) < 4:
            return -1
        return struct.unpack("<I", buf)[0]

    def check_block(self, want: str) -> bool:
        peek = self._peek_tag()
        if peek < 0:
            return False
        return int(tags[want].value, 16) == peek

    def next_block(self, want: str) -> int:
        """Scan forward to block. Leave cursor at tag start. Return payload size or -1."""
        want_id = int(tags[want].value, 16)
        start = self.f.tell()
        while True:
            hdr = self.f.read(8)
            if not hdr:
                self.f.seek(start)
                return -1
            tag_id, sz = struct.unpack("<II", hdr)
            if tag_id == want_id:
                self.f.seek(-8, os.SEEK_CUR)
                return sz
            self.f.seek(sz, os.SEEK_CUR)

    def enter(self, want: str) -> int:
        """Enter a block and return its payload size. Cursor at payload start."""
        tag_id = self.read_u32()
        if tag_id != int(tags[want].value, 16):
            raise RuntimeError(f"Expected block {want}")
        return self.read_u32()

    def skip(self, n: int) -> None:
        self.f.seek(n, os.SEEK_CUR)


# ---------------------------------------------------------------------
# Results model with chainable selectors and pretty printing
# ---------------------------------------------------------------------

@dataclass
class _Field:
    """Result variable across domains and time."""
    name: str
    fmt: str                  # "NODE" | "ITEM" | "SURFACE"
    comp: int                 # components per item
    times: np.ndarray         # (T,)
    data: List[np.ndarray]    # per-domain: (T, n_items, comp)
    gidx: List[np.ndarray]    # per-domain: global ids
    domains: List[str]        # per-domain names

    def last(self) -> "_View":
        return _View(self, -1, None)

    def at(self, tsel: int | slice | Sequence[int]) -> "_View":
        return _View(self, tsel, None)

    def domain(self, dsel: str | int | Sequence[str | int]) -> "_View":
        return _View(self, slice(None), dsel)

    def __getitem__(self, t: int | slice) -> "_View":
        return self.at(t)

    def __repr__(self) -> str:
        T = int(self.times.size)
        doms = ", ".join(f"{nm}:{self.data[i].shape[1]}" for i, nm in enumerate(self.domains))
        return f"<Field {self.name!r} fmt={self.fmt} comp={self.comp} T={T} domains=[{doms}]>"


class Results(Mapping[str, _Field]):
    """Dictionary-like results container with nice summaries."""

    def __init__(self, fields: Dict[str, _Field] | None = None) -> None:
        self._fields: Dict[str, _Field] = {} if fields is None else dict(fields)

    # Mapping interface
    def __getitem__(self, key: str) -> _Field:
        return self._fields[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    # Mutator for internal use
    def _set(self, key: str, value: _Field) -> None:
        self._fields[key] = value

    def keys(self) -> Iterable[str]:  # type: ignore[override]
        return self._fields.keys()

    def __repr__(self) -> str:
        lines = ["<Results:"]
        for k, f in self._fields.items():
            T = int(f.times.size)
            ndoms = len(f.domains)
            lines.append(f"  - {k}  fmt={f.fmt}  comp={f.comp}  T={T}  domains={ndoms}")
        lines.append(">")
        return "\n".join(lines)


@dataclass
class _View:
    """Time and domain selection view."""
    field: _Field
    tsel: int | slice | Sequence[int]
    dsel: str | int | Sequence[str | int] | None

    def _domain_indices(self) -> List[int]:
        if self.dsel is None:
            return list(range(len(self.field.data)))
        if isinstance(self.dsel, (str, int)):
            s = str(self.dsel)
            return [i for i, nm in enumerate(self.field.domains) if nm == s or str(i) == s]
        want = {str(x) for x in self.dsel}
        return [i for i, nm in enumerate(self.field.domains) if nm in want or str(i) in want]

    def _stack(self) -> Tuple[np.ndarray, np.ndarray]:
        didx = self._domain_indices()
        chunks, gids = [], []
        for i in didx:
            chunks.append(self.field.data[i][self.tsel])  # (T?, n_items, comp) or (n_items, comp)
            gids.append(self.field.gidx[i])
        return np.concatenate(chunks, axis=-2), np.concatenate(gids, axis=0)

    def nodal(self, mesh: Mesh) -> np.ndarray:
        if self.field.fmt != "NODE":
            raise ValueError("Not a nodal field")
        vals, gids = self._stack()
        N = mesh.nnodes
        if vals.ndim == 2:
            out = np.zeros((N, vals.shape[-1]), vals.dtype); out[gids] = vals; return out
        T = vals.shape[0]
        out = np.zeros((T, N, vals.shape[-1]), vals.dtype); out[:, gids, :] = vals; return out

    def elements(self, mesh: Mesh) -> np.ndarray:
        if self.field.fmt != "ITEM":
            raise ValueError("Not an element field")
        vals, gids = self._stack()
        E = mesh.nelems
        if vals.ndim == 2:
            out = np.zeros((E, vals.shape[-1]), vals.dtype); out[gids] = vals; return out
        T = vals.shape[0]
        out = np.zeros((T, E, vals.shape[-1]), vals.dtype); out[:, gids, :] = vals; return out

    def raw(self) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for i in self._domain_indices():
            out.append(self.field.data[i][self.tsel])
        return out


# ---------------------------------------------------------------------
# XPLT reader
# ---------------------------------------------------------------------

class Xplt:
    """Efficient FEBio XPLT reader, compatible with the new Mesh."""

    path: str
    version: int
    compression: int
    dictionary: Dict[str, Dict[str, str]]
    results: Results
    mesh: Mesh
    times: np.ndarray

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self.dictionary = {}
        self.results = Results()
        self.times = np.empty((0,), dtype=float)
        self._read_all(self.path)

    def __getitem__(self, var: str) -> _Field:
        return self.results[var]

    def __repr__(self) -> str:
        nvars = len(self.results)
        t = int(self.times.size)
        nn = getattr(self, "mesh", None).nnodes if hasattr(self, "mesh") else 0
        ne = getattr(self, "mesh", None).nelems if hasattr(self, "mesh") else 0
        return f"<Xplt file={os.path.basename(self.path)!r} ver=0x{self.version:04x} times={t} nodes={nn} elems={ne} vars={nvars}>"

    # ---------------- pipeline ----------------

    def _read_all(self, path: str) -> None:
        br = _Bin(path)

        if br.read_u32() != 4605250:
            br.close()
            raise RuntimeError("Not a valid XPLT file")

        br.next_block("PLT_ROOT");            br.enter("PLT_ROOT")
        br.next_block("PLT_HEADER");          br.enter("PLT_HEADER")
        br.next_block("PLT_HDR_VERSION");     br.enter("PLT_HDR_VERSION");   self.version = br.read_u32()
        br.next_block("PLT_HDR_COMPRESSION"); br.enter("PLT_HDR_COMPRESSION"); self.compression = br.read_u32()

        self._read_dictionary(br)
        self.mesh = self._read_mesh(br)
        self._read_parts(br)
        self._read_states(br)

        br.close()

    # ---------------- dictionary --------------

    def _read_dictionary(self, br: _Bin) -> None:
        br.next_block("PLT_DICTIONARY"); br.enter("PLT_DICTIONARY")
        self.dictionary = {}

        def _stream(tag: str) -> None:
            if br.next_block(tag) < 0:
                return
            br.enter(tag)
            while br.check_block("PLT_DIC_ITEM"):
                br.enter("PLT_DIC_ITEM")
                br.next_block("PLT_DIC_ITEM_TYPE"); br.enter("PLT_DIC_ITEM_TYPE"); itype = br.read_u32()
                br.next_block("PLT_DIC_ITEM_FMT");  br.enter("PLT_DIC_ITEM_FMT");  ifmt  = br.read_u32()
                br.next_block("PLT_DIC_ITEM_NAME"); br.enter("PLT_DIC_ITEM_NAME")
                name = br.read_bytes(64).decode("utf-8", "ignore").split("\x00")[0]
                self.dictionary[name] = {"type": FEDataType(itype).name, "format": Storage_Fmt(ifmt).name}

        _stream("PLT_DIC_NODAL")
        _stream("PLT_DIC_DOMAIN")
        _stream("PLT_DIC_SURFACE")

        for k, meta in self.dictionary.items():
            comp = FEDataDim[meta["type"]].value
            self.results._set(k, _Field(
                name=k,
                fmt=meta["format"],
                comp=comp,
                times=np.empty((0,), dtype=float),
                data=[],
                gidx=[],
                domains=[],
            ))

    # ---------------- mesh --------------------

    @staticmethod
    def _etype_str(etype: str) -> str:
        s = etype.upper().removeprefix("FE_")
        for p in ("TET", "HEX", "PENTA", "TRI", "QUAD", "LINE"):
            if s.startswith(p):
                return p.lower() + s[len(p):].lower()
        return s.lower()

    def _read_mesh(self, br: _Bin) -> Mesh:
        br.next_block("PLT_MESH"); br.enter("PLT_MESH")

        # Nodes
        br.next_block("PLT_NODE_SECTION"); br.enter("PLT_NODE_SECTION")
        br.next_block("PLT_NODE_HEADER");  br.enter("PLT_NODE_HEADER")
        br.next_block("PLT_NODE_SIZE");    br.enter("PLT_NODE_SIZE"); nnode = br.read_u32()
        br.next_block("PLT_NODE_DIM");     br.enter("PLT_NODE_DIM");  ndim  = br.read_u32()
        br.next_block("PLT_NODE_COORDS");  br.enter("PLT_NODE_COORDS")

        ids = np.empty((nnode,), dtype=np.int64)
        xyz = np.zeros((nnode, max(3, ndim)), dtype=np.float32)
        for i in range(nnode):
            ids[i] = br.read_u32()
            xyz[i, 0] = br.read_f32()
            xyz[i, 1] = br.read_f32()
            xyz[i, 2] = br.read_f32() if ndim > 2 else 0.0

        order = np.argsort(ids)
        remap = np.empty(ids.size, dtype=np.int64); remap[order] = np.arange(ids.size)
        nodes = NodeArray(xyz[order])

        # Domains and elements
        parts: Dict[str, List[int]] = {}
        conn_list: List[np.ndarray] = []
        etypes: List[str] = []

        if br.next_block("PLT_DOMAIN_SECTION") >= 0:
            br.enter("PLT_DOMAIN_SECTION")
            while br.check_block("PLT_DOMAIN"):
                br.enter("PLT_DOMAIN")
                br.next_block("PLT_DOMAIN_HDR"); br.enter("PLT_DOMAIN_HDR")
                br.next_block("PLT_DOM_ELEM_TYPE"); br.enter("PLT_DOM_ELEM_TYPE"); et = Elem_Type(br.read_u32()).name
                br.next_block("PLT_DOM_PART_ID");  br.enter("PLT_DOM_PART_ID");  pid = br.read_u32()
                br.next_block("PLT_DOM_ELEMS");    br.enter("PLT_DOM_ELEMS");    nelem = br.read_u32()
                nlen = br.next_block("PLT_DOM_NAME")
                dname = None
                if nlen > 0:
                    br.enter("PLT_DOM_NAME")
                    dname = br.read_bytes(nlen).decode("utf-8", "ignore").split("\x00")[-1]

                br.next_block("PLT_DOM_ELEM_LIST"); br.enter("PLT_DOM_ELEM_LIST")
                for _ in range(nelem):
                    sz = br.enter("PLT_ELEMENT")
                    raw = np.frombuffer(br.read_bytes(sz), dtype="<u4")
                    conn = remap[raw[1:].astype(np.int64) - 1]
                    conn_list.append(conn)
                    etypes.append(self._etype_str(et))
                parts.setdefault(dname or str(pid), []).append(len(etypes) - 1)

        # Elements table (NumPy)
        kmax = max((len(c) for c in conn_list), default=0)
        conn = -np.ones((len(conn_list), kmax), dtype=np.int64)
        nper = np.zeros((len(conn_list),), dtype=np.int64)
        for i, c in enumerate(conn_list):
            conn[i, : c.size] = c
            nper[i] = c.size
        elements = ElementArray(conn=conn, nper=nper, etype=np.asarray(etypes, dtype=object))
        parts_idx = {k: np.asarray(v, dtype=np.int64) for k, v in parts.items()}

        # Surfaces (optional)
        surfaces: Dict[str, SurfaceArray] = {}
        if br.next_block("PLT_SURFACE_SECTION") >= 0:
            br.enter("PLT_SURFACE_SECTION")
            while br.check_block("PLT_SURFACE"):
                br.enter("PLT_SURFACE")
                br.next_block("PLT_SURFACE_HDR"); br.enter("PLT_SURFACE_HDR")
                br.next_block("PLT_SURFACE_ID");  br.enter("PLT_SURFACE_ID");  sid = br.read_u32()
                br.next_block("PLT_SURFACE_FACES"); br.enter("PLT_SURFACE_FACES"); nfaces = br.read_u32()
                nlen = br.next_block("PLT_SURFACE_NAME")
                sname = None
                if nlen > 0:
                    br.enter("PLT_SURFACE_NAME")
                    sname = br.read_bytes(nlen).decode("utf-8", "ignore").split("\x00")[-1]
                br.next_block("PLT_SURFACE_MAX_FACET_NODES"); br.enter("PLT_SURFACE_MAX_FACET_NODES"); kfac = br.read_u32()

                faces_list: List[np.ndarray] = []
                if br.next_block("PLT_FACE_LIST") >= 0:
                    br.enter("PLT_FACE_LIST")
                    for _ in range(nfaces):
                        sz = br.enter("PLT_FACE")
                        raw = np.frombuffer(br.read_bytes(sz), dtype="<u4")
                        if raw.size >= 2 + kfac:
                            nn = remap[raw[2 : 2 + kfac].astype(np.int64) - 1]
                            faces_list.append(nn)
                if faces_list:
                    kk = max(len(a) for a in faces_list)
                    F = len(faces_list)
                    fa = -np.ones((F, kk), dtype=np.int64)
                    nps = np.zeros((F,), dtype=np.int64)
                    for i, a in enumerate(faces_list):
                        fa[i, : a.size] = a
                        nps[i] = a.size
                    surfaces[sname or f"surface_{sid}"] = SurfaceArray(faces=fa, nper=nps)

        # Nodesets (optional)
        nodesets: Dict[str, np.ndarray] = {}
        if br.next_block("PLT_NODESET_SECTION") >= 0:
            br.enter("PLT_NODESET_SECTION")
            while br.check_block("PLT_NODESET"):
                br.enter("PLT_NODESET")
                br.next_block("PLT_NODESET_HDR");  br.enter("PLT_NODESET_HDR")
                br.next_block("PLT_NODESET_ID");   br.enter("PLT_NODESET_ID");   nid = br.read_u32()
                br.next_block("PLT_NODESET_SIZE"); br.enter("PLT_NODESET_SIZE"); n = br.read_u32()
                nlen = br.next_block("PLT_NODESET_NAME")
                nm = None
                if nlen > 0:
                    br.enter("PLT_NODESET_NAME")
                    nm = br.read_bytes(nlen).decode("utf-8", "ignore").split("\x00")[-1]
                if br.next_block("PLT_NODESET_LIST") >= 0:
                    br.enter("PLT_NODESET_LIST")
                    raw = np.frombuffer(br.read_bytes(4 * n), dtype="<u4")
                    nodesets[nm or f"nodeset_{nid}"] = remap[raw.astype(np.int64) - 1]

        return Mesh(nodes=nodes, elements=elements, parts=parts_idx, surfaces=surfaces, nodesets=nodesets)

    # --------------- parts --------------------

    def _read_parts(self, br: _Bin) -> None:
        if br.next_block("PLT_PARTS_SECTION") < 0:
            return
        br.enter("PLT_PARTS_SECTION")
        # Names already captured from domains.

    # --------------- states/results -----------

    def _read_states(self, br: _Bin) -> None:
        times: List[float] = []
        # tmp[name] -> list of (domain_name, [time_slices as np.ndarray])
        tmp: Dict[str, List[Tuple[str, List[np.ndarray]]]] = {k: [] for k in self.results.keys()}

        dom_names = list(self.mesh.parts.keys())
        item_gidx = {dn: self.mesh.parts[dn] for dn in dom_names}
        nodal_gidx = {dn: np.unique(self.mesh.elements[self.mesh.parts[dn]].unique_nodes()) for dn in dom_names}

        while True:
            if br.next_block("PLT_STATE") < 0:
                break
            br.enter("PLT_STATE")

            br.next_block("PLT_STATE_HEADER"); br.enter("PLT_STATE_HEADER")
            br.next_block("PLT_STATE_HDR_TIME"); br.enter("PLT_STATE_HDR_TIME")
            times.append(br.read_f32())
            br.next_block("PLT_STATE_STATUS"); br.enter("PLT_STATE_STATUS")
            status = br.read_u32()

            sz_state = br.next_block("PLT_STATE_DATA")
            if sz_state < 0:
                continue
            br.enter("PLT_STATE_DATA")
            state_end = br.tell() + sz_state
            if status != 0:
                br.seek(state_end); continue

            # NODE stream
            self._var_cursor = 0
            pos = br.next_block("PLT_NODE_DATA")
            if pos >= 0 and br.tell() < state_end:
                sz = br.enter("PLT_NODE_DATA")
                self._read_state_stream(br, tmp, dom_names, is_node=True, stream_end=br.tell() + sz)

            # ELEMENT stream
            self._var_cursor = 0
            pos = br.next_block("PLT_ELEMENT_DATA")
            if pos >= 0 and br.tell() < state_end:
                sz = br.enter("PLT_ELEMENT_DATA")
                self._read_state_stream(br, tmp, dom_names, is_node=False, stream_end=br.tell() + sz)

            br.seek(state_end)

        self.times = np.asarray(times, dtype=float)
        for name, fld in self.results._fields.items():
            fld.times = self.times
            dom_series = tmp.get(name, [])
            fld.data, fld.gidx, fld.domains = [], [], []
            if not dom_series:
                continue
            for dname, series in dom_series:
                # stack to NumPy once, no lists left afterward
                arr = np.stack(series, axis=0) if series else np.zeros((len(self.times), 0, fld.comp), dtype=np.float32)
                fld.data.append(arr)
                fld.domains.append(dname)
                if fld.fmt == "ITEM":
                    fld.gidx.append(item_gidx.get(dname, np.arange(arr.shape[1], dtype=np.int64)))
                elif fld.fmt == "NODE":
                    fld.gidx.append(nodal_gidx.get(dname, np.arange(arr.shape[1], dtype=np.int64)))
                else:
                    fld.gidx.append(np.arange(arr.shape[1], dtype=np.int64))

    def _read_state_stream(
        self,
        br: _Bin,
        tmp: Dict[str, List[Tuple[str, List[np.ndarray]]]],
        dom_names: List[str],
        is_node: bool,
        stream_end: int,
    ) -> None:
        while br.tell() < stream_end:
            if not br.check_block("PLT_STATE_VARIABLE"):
                break
            br.enter("PLT_STATE_VARIABLE")
            br.next_block("PLT_STATE_VAR_ID"); br.enter("PLT_STATE_VAR_ID"); _ = br.read_u32()

            dict_key = list(self.dictionary.keys())[self._var_cursor]
            comp = FEDataDim[self.dictionary[dict_key]["type"]].value

            var_sz = br.next_block("PLT_STATE_VAR_DATA")
            if var_sz < 0:
                break
            br.enter("PLT_STATE_VAR_DATA")
            var_end = br.tell() + var_sz

            while br.tell() + 8 <= var_end:
                dom_id = br.read_u32()
                dom_idx = dom_id - 1
                sz = br.read_u32()
                if sz == 0 or br.tell() + sz > var_end:
                    br.seek(min(var_end, br.tell() + sz)); continue
                buf = br.read_bytes(sz)
                fcnt = len(buf) // 4
                if fcnt == 0 or comp == 0:
                    continue
                n_items = fcnt // comp
                if n_items == 0:
                    continue
                arr = np.frombuffer(buf, dtype="<f4", count=n_items * comp).reshape(n_items, comp)

                dname = dom_names[dom_idx] if 0 <= dom_idx < len(dom_names) else str(dom_id)
                bucket = None
                for j, (nm, series) in enumerate(tmp[dict_key]):
                    if nm == dname:
                        bucket = series; break
                if bucket is None:
                    tmp[dict_key].append((dname, []))
                    bucket = tmp[dict_key][-1][1]
                bucket.append(arr.astype(np.float32, copy=False))

            br.seek(var_end)
            self._var_cursor += 1
