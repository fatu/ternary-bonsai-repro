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

    def get(self, name, float32: bool = False) -> np.ndarray:
        h = self.header[name]
        a, b = h["data_offsets"]
        arr = self._mm[self._base + a:self._base + b].view(_NP[h["dtype"]]).reshape(h["shape"])
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

    def get(self, name, float32: bool = False) -> np.ndarray:
        return self.where[name].get(name, float32)
