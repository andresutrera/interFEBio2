# ---------------------------------------------------------------------
# Binary reader (fast)
# ---------------------------------------------------------------------
from __future__ import annotations

import mmap
import os
import struct

from .Enums import tags


class BinaryReader:
    """TLV reader optimized for FEBio .xplt.

    Design
    ------
    - Memory-mapped file, zero-copy slices.
    - Iterative tag scan (no recursion).
    - Integer cursor `pos` replaces file seeks.
    - Backward compatible API: `read`, `search_block`, `check_block`, `seek_block`.

    Notes
    -----
    Tag records are stored as little-endian:
        [u32 tag][u32 size][size bytes of payload]
    """

    __slots__ = ("_f", "_mm", "_buf", "filesize", "pos", "_tag2int")

    def __init__(self, filename: str):
        f = open(filename, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._f = f
        self._mm = mm
        self._buf = memoryview(mm)
        self.filesize = mm.size()
        self.pos = 0
        # Cache: str tag -> int code
        self._tag2int = {
            name: int(enum.value, 16) for name, enum in tags.__members__.items()
        }

    def __repr__(self) -> str:
        return f"BinaryReader(size={self.filesize}, pos={self.pos})"

    __str__ = __repr__

    # ------------- low-level cursor ops -------------

    def _ensure(self, n: int) -> None:
        if self.pos + n > self.filesize:
            raise EOFError("read past end of file")

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> None:
        if whence == os.SEEK_SET:
            np = offset
        elif whence == os.SEEK_CUR:
            np = self.pos + offset
        elif whence == os.SEEK_END:
            np = self.filesize + offset
        else:
            raise ValueError("invalid whence")
        if np < 0 or np > self.filesize:
            raise ValueError("invalid seek")
        self.pos = np

    def skip(self, n: int) -> None:
        self._ensure(n)
        self.pos += n

    # ------------- typed reads -------------

    def read(self, n: int = 4) -> bytes:
        """Return `n` bytes and advance."""
        self._ensure(n)
        out = self._buf[self.pos : self.pos + n].tobytes()
        self.pos += n
        return out

    def read_u32(self) -> int:
        """Read little-endian unsigned 32-bit."""
        self._ensure(4)
        val = struct.unpack_from("<I", self._buf, self.pos)[0]
        self.pos += 4
        return int(val)

    def peek_u32(self) -> int:
        """Peek next u32 without advancing."""
        self._ensure(4)
        return int(struct.unpack_from("<I", self._buf, self.pos)[0])

    def read_f32(self) -> float:
        self._ensure(4)
        val = struct.unpack_from("<f", self._buf, self.pos)[0]
        self.pos += 4
        return float(val)

    # ------------- tag helpers -------------

    def _tagcode(self, BLOCK_TAG: str) -> int:
        try:
            return self._tag2int[BLOCK_TAG]
        except KeyError:
            # Fallback without cache; should not happen
            return int(tags[BLOCK_TAG].value, 16)

    def search_block(
        self,
        BLOCK_TAG: str,
        max_depth: int = 5,  # kept for signature compatibility; unused
        cur_depth: int = 0,  # kept for signature compatibility; unused
        print_tag: int = 0,  # kept for signature compatibility; unused
    ) -> int:
        """Scan forward for the next tag. Return payload size or -1.

        Leaves cursor at the *payload start* of the found block.
        Restores position if not found.
        """
        want = self._tagcode(BLOCK_TAG)
        start = self.pos
        p = self.pos
        end = self.filesize

        # Need at least 8 bytes for a header
        while p + 8 <= end:
            tag = struct.unpack_from("<I", self._buf, p)[0]
            size = struct.unpack_from("<I", self._buf, p + 4)[0]
            payload = p + 8
            next_rec = payload + size
            if next_rec > end:
                # Corrupt size. Abort and restore.
                self.pos = start
                return -1
            if tag == want:
                # Position cursor at payload
                self.pos = payload
                return int(size)
            p = next_rec

        # Not found; restore
        self.pos = start
        return -1

    def check_block(self, BLOCK_TAG: str, filesize: int = -1) -> int:
        """Return 1 if next header's tag equals BLOCK_TAG. Cursor unchanged."""
        if filesize > 0 and self.pos + 4 > filesize:
            return 0
        try:
            return 1 if self.peek_u32() == self._tagcode(BLOCK_TAG) else 0
        except EOFError:
            return 0

    def seek_block(self, BLOCK_TAG: str) -> int:
        """Consume the next header if it matches. Return payload size.

        Cursor is left at the payload start.
        """
        tag = self.read_u32()
        size = self.read_u32()
        if tag != self._tagcode(BLOCK_TAG):
            # Undo and signal mismatch by rewinding 8 bytes for safety
            self.pos -= 8
            raise ValueError(f"seek_block expected {BLOCK_TAG}, found tag=0x{tag:08x}")
        return int(size)

    # ------------- lifecycle -------------

    def close(self) -> None:
        try:
            self._buf.release()
        except Exception:
            pass
        try:
            self._mm.close()
        finally:
            self._f.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
