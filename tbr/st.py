"""Minimal safetensors reader: header parse + zero-copy memmap views. No torch, no MLX.

Covers the dtypes in both checkpoints: U32 (MLX packed codes), F16 (scales, biases, signs,
FP-kept tensors), BF16 (the Qwen base; returned as float32), F32.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_NP = {"U8": np.uint8, "I8": np.int8, "U16": np.uint16, "I16": np.int16, "U32": np.uint32,
       "I32": np.int32, "I64": np.int64, "F16": np.float16, "BF16": np.uint16, "F32": np.float32}


def read_header(path) -> tuple[dict, dict, int]:
    """(tensor header, __metadata__, byte offset where tensor data starts)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, header.pop("__metadata__", {}) or {}, 8 + n


def expected_size(path) -> int:
    """File size the header implies; a truncated download is smaller than this."""
    header, _, base = read_header(path)
    return base + max((h["data_offsets"][1] for h in header.values()), default=0)


class SafeTensors:
    def __init__(self, path):
        self.path = Path(path)
        self.header, self.metadata, self._base = read_header(self.path)
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")

    def keys(self):
        return self.header.keys()

    def info(self, name) -> dict:
        return self.header[name]

    def get(self, name, float32: bool = False, rows=None) -> np.ndarray:
        """Whole tensor, or `rows` (a slice or an index array) of it — only those bytes are touched."""
        h = self.header[name]
        a, b = h["data_offsets"]
        arr = self._mm[self._base + a:self._base + b].view(_NP[h["dtype"]]).reshape(h["shape"])
        if rows is not None:
            arr = arr[rows]
        if h["dtype"] == "BF16":
            return (arr.astype(np.uint32) << 16).view(np.float32)
        return arr.astype(np.float32) if float32 else arr


class Checkpoint:
    """Every *.safetensors file in a directory, addressed by tensor name."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.files = [SafeTensors(p) for p in sorted(self.dir.glob("*.safetensors"))]
        if not self.files:
            raise FileNotFoundError(f"no *.safetensors in {self.dir}")
        self.where = {k: f for f in self.files for k in f.keys()}

    def keys(self):
        return self.where.keys()

    def info(self, name) -> dict:
        return self.where[name].info(name)

    def get(self, name, float32: bool = False, rows=None) -> np.ndarray:
        return self.where[name].get(name, float32, rows)


_ST = {np.dtype(np.uint8): "U8", np.dtype(np.uint16): "U16", np.dtype(np.uint32): "U32", np.dtype(np.int8): "I8",
       np.dtype(np.float16): "F16", np.dtype(np.float32): "F32"}


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit pattern (uint16, truncation), what a BF16 safetensors tensor stores."""
    return (np.ascontiguousarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def write_safetensors(path, tensors: dict, bf16=(), metadata=None):
    """Write one safetensors file. Arrays named in `bf16` must be uint16 bit patterns (see bf16_bits)."""
    header, blobs, off = {}, [], 0
    for name, a in tensors.items():
        a = np.ascontiguousarray(a)
        raw = a.tobytes()
        header[name] = {"dtype": "BF16" if name in bf16 else _ST[a.dtype], "shape": list(a.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    if metadata:
        header["__metadata__"] = metadata
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs:
            f.write(b)
