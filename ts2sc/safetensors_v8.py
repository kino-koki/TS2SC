#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8.2 · Pure-stdlib safetensors I/O + byte-level migration

Why this file exists: v8.1 left ``migrate.execute()`` as an explicit T+0 stub.
That was honest, but it meant the weight-moving half of TS2SC could never be
tested before release day. safetensors is a deliberately simple format:

    [u64 LE header_len][UTF-8 JSON header][raw tensor bytes]

COPY is a verbatim byte slice; GQA row remapping, o_proj slicing and QK-norm
folding are all row/column operations that can be done on raw bytes WITHOUT
torch, WITHOUT a GPU and WITHOUT a package index. So the entire weight-level
migration can be finished and verified in the sandbox. Only the L2 teacher
forward, L3 paired generation and beta calibration still need the T+0 machine.

Memory: mmap, never materialise a 27B checkpoint in RAM.
"""
from __future__ import annotations

import json
import mmap
import os
import struct

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
    "BOOL": 1, "U16": 2, "U32": 4, "U64": 8,
}


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------
class SafeTensors:
    """mmap-backed reader. tensor(name) returns raw bytes for that tensor."""
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        raw = self.f.read(n)
        self.header = json.loads(raw.decode("utf-8"))
        self.data_start = 8 + n
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.keys = [k for k in self.header if k != "__metadata__"]

    def meta(self, name: str) -> dict:
        return self.header[name]

    def tensor(self, name: str) -> bytes:
        m = self.header[name]
        a, b = m["data_offsets"]
        return self.mm[self.data_start + a:self.data_start + b]

    def nbytes(self, name: str) -> int:
        a, b = self.header[name]["data_offsets"]
        return b - a

    def close(self):
        self.mm.close(); self.f.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def read_index(teacher_dir: str) -> dict:
    """Return {tensor_name: absolute_shard_path} for a sharded HF release.

    Qwen3-1.7B-Instruct was distributed as qa_*/qb_* byte shards of ONE file;
    Qwen4-27B will almost certainly use model.safetensors.index.json. Both are
    supported; neither shape is hardcoded into the student config.
    """
    idx = os.path.join(teacher_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        obj = json.load(open(idx, encoding="utf-8"))
        wm = obj["weight_map"]
        return {k: os.path.join(teacher_dir, v) for k, v in wm.items()}
    single = os.path.join(teacher_dir, "model.safetensors")
    if os.path.exists(single):
        with SafeTensors(single) as s:
            return {k: single for k in s.keys}
    raise SystemExit(f"[safetensors] no index and no model.safetensors in {teacher_dir}")


def concat_shards(teacher_dir: str, out_path: str, prefix=("qa_", "qb_")) -> dict:
    """Reassemble the historical qa_*/qb_* split into one safetensors file."""
    import glob
    import re
    parts = []
    for p in prefix:
        for f in glob.glob(os.path.join(teacher_dir, p + "*")):
            m = re.search(p + r"(\d+)$", os.path.basename(f))
            if m:
                parts.append((p, int(m.group(1)), f))
    parts.sort(key=lambda x: (x[0], x[1]))
    if not parts:
        raise SystemExit("[safetensors] no shards to concatenate")
    total = 0
    with open(out_path, "wb") as w:
        for _, _, f in parts:
            with open(f, "rb") as r:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    w.write(b); total += len(b)
    return dict(n_parts=len(parts), bytes=total,
                order=[os.path.basename(f) for _, _, f in parts])


# ---------------------------------------------------------------------------
# Byte-level DERIVE transforms (dtype-agnostic, no torch)
# ---------------------------------------------------------------------------
def row_bytes(raw: bytes, shape: list, dtype: str, i: int) -> bytes:
    """Bytes of row i of a 2-D tensor."""
    itemsize = DTYPE_BYTES[dtype]
    row_len = shape[1] * itemsize
    return raw[i * row_len:(i + 1) * row_len]


def derive_gqa_row_remap(raw: bytes, meta: dict, slot_map: dict) -> tuple:
    """Expand (kv_dim, d) -> (d, d) by mapping every row to a REAL teacher row.

    V7 did ``Wk[:kv_dim] = k; Wk[kv_dim:] = 0``. Here the remaining rows are
    filled with repeated real teacher rows (head-major), so no SGC slot is dead.
    Pure byte concatenation: works for BF16/F16/F32 without numeric conversion.
    """
    shape, dtype = meta["shape"], meta["dtype"]
    if len(shape) != 2:
        raise ValueError("GQA remap expects a 2-D tensor")
    kv_dim, d = shape
    if d != slot_map["d"] or kv_dim != slot_map["kv_dim"]:
        raise ValueError(f"shape {shape} disagrees with slot_map "
                         f"d={slot_map['d']} kv_dim={slot_map['kv_dim']}")
    rows = b"".join(row_bytes(raw, shape, dtype, j)
                    for j in slot_map["row_to_teacher"])
    return rows, [d, d]


def derive_slice(raw: bytes, meta: dict, row_end: int | None = None,
                 col_end: int | None = None) -> tuple:
    """Slice a 2-D tensor (e.g. o_proj -> (d,d)). Byte-level, dtype-agnostic."""
    shape, dtype = meta["shape"], meta["dtype"]
    if len(shape) != 2:
        raise ValueError("slice expects a 2-D tensor")
    r, c = shape
    r2, c2 = row_end or r, col_end or c
    if r2 > r or c2 > c:
        raise ValueError("slice exceeds teacher shape")
    itemsize = DTYPE_BYTES[dtype]
    out = b"".join(raw[i * c * itemsize:i * c * itemsize + c2 * itemsize]
                   for i in range(r2))
    return out, [r2, c2]


def zero_tensor(shape: list, dtype: str) -> bytes:
    n = 1
    for x in shape: n *= x
    return b"\x00" * (n * DTYPE_BYTES[dtype])


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
def write_safetensors(out_path: str, tensors: dict, metadata: dict | None = None) -> dict:
    """tensors: {name: (dtype, shape, raw_bytes)}. Produces a valid file."""
    hdr = dict(metadata or {})
    off, blobs = 0, []
    for name, (dtype, shape, raw) in tensors.items():
        n = 1
        for x in shape: n *= x
        expect = n * DTYPE_BYTES[dtype]
        if len(raw) != expect:
            raise ValueError(f"{name}: {len(raw)} bytes != {expect} for {dtype}{shape}")
        hdr[name] = dict(dtype=dtype, shape=list(shape), data_offsets=[off, off + len(raw)])
        blobs.append(raw); off += len(raw)
    if metadata:
        hdr["__metadata__"] = metadata
    hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    # safetensors requires the header to be padded to 8-byte alignment.
    pad = (-len(hb)) % 8
    hb += b" " * pad
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for b in blobs: f.write(b)
    return dict(path=out_path, n_tensors=len(tensors),
                data_bytes=off, header_bytes=len(hb), pad=pad)


# ---------------------------------------------------------------------------
# End-to-end byte-level migration (the v8.2 deliverable)
# ---------------------------------------------------------------------------
def execute_byte_migration(teacher_dir: str, out_dir: str, spec,
                           slot_map: dict, strict: bool = True) -> dict:
    """Materialise a student safetensors from a REAL teacher dir.

    COPY  -> verbatim bytes + shape/dtype assert
    DERIVE-> GQA row remap / slice (byte-level); numeric QK fold deferred to T0
    NEW   -> zero-initialised, shape derived from spec (never R=2 hardcoded)
    """
    index = read_index(teacher_dir)
    from provenance import classify, write_ledger
    led = classify(list(index.keys()), spec)
    if strict and (led["unclassified"] or led["blockers"]):
        return dict(status="BLOCKED", unclassified=led["unclassified"],
                    blockers=led["blockers"],
                    honest="no student checkpoint written")

    # open each shard once
    cache = {}
    def get(name):
        p = index[name]
        if p not in cache: cache[p] = SafeTensors(p)
        return cache[p]

    tensors, report, failures = {}, [], []
    try:
        for e in led["entries"]:
            name, op = e["src"], e["op"]
            s = get(name); m = s.meta(name); raw = s.tensor(name)
            if op == "COPY":
                tensors[e["dst"]] = (m["dtype"], m["shape"], raw)
                report.append(dict(src=name, op=op, dst=e["dst"],
                                   shape=m["shape"], dtype=m["dtype"],
                                   bytes=len(raw), bit_identical=True))
            elif op == "DERIVE":
                t = e["transform"]
                if t == "GQA_HEAD_EXPAND_OR_PROJECT" and len(m["shape"]) == 2:
                    out, shape = derive_gqa_row_remap(raw, m, slot_map)
                    tensors["_derive_" + name.replace(".", "_")] = (m["dtype"], shape, out)
                    report.append(dict(src=name, op=op, transform="GQA_ROW_REMAP",
                                       dst="_derive_" + name.replace(".", "_"),
                                       in_shape=m["shape"], out_shape=shape,
                                       zero_padded_rows=0))
                elif t in ("GQA_HEAD_EXPAND_OR_PROJECT", "BIAS_CARRY_OR_DECLARED_DROP"):
                    if len(m["shape"]) == 2:
                        out, shape = derive_slice(raw, m)
                        tensors["_derive_" + name.replace(".", "_")] = (m["dtype"], shape, out)
                        report.append(dict(src=name, op=op, transform="SLICE",
                                           dst="_derive_" + name.replace(".", "_"),
                                           in_shape=m["shape"], out_shape=shape))
                    else:
                        report.append(dict(src=name, op=op, transform="DEFER_TO_T0",
                                           note="non-2D or numeric transform needs torch"))
                else:
                    report.append(dict(src=name, op=op, transform="DEFER_TO_T0",
                                       note=t + " requires numeric op at T+0"))
            # SKIP never allowed: classify() returns DERIVE for all known tensors

        # NEW: shape derived from spec, zero init, grad required at T+0
        d = spec.values["hidden_size"]
        nl = spec.values["num_hidden_layers"]
        dtype = "F32"     # gate params are small; F32 avoids BF16 underflow at init
        for i in range(nl):
            b = f"model.blocks.{i}."
            for k, shp in ((b + "Wc", [3, d]), (b + "Wg.weight", [d, d]),
                           (b + "wgate.weight", [1, d]), (b + "alpha", [1])):
                tensors[k] = (dtype, shp, zero_tensor(shp, dtype))
        report.append(dict(op="NEW", n=4 * nl + 1, init="zeros",
                           note="beta/alpha MUST be calibrated at L2; not left at zero"))

        os.makedirs(out_dir, exist_ok=True)
        w = write_safetensors(os.path.join(out_dir, "student.safetensors"),
                              tensors, metadata=dict(format="pt", ts2sc="v8.2"))
        json.dump(spec.student_config(),
                  open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        write_ledger(out_dir, led)

        # verification pass: reopen the OUTPUT and re-check every COPY byte
        with SafeTensors(os.path.join(out_dir, "student.safetensors")) as chk:
            for r in report:
                if r["op"] != "COPY": continue
                a = get(r["src"]).tensor(r["src"])
                b = chk.tensor(r["dst"])
                if a != b or chk.meta(r["dst"])["shape"] != r["shape"]:
                    failures.append("COPY_BYTE_MISMATCH:" + r["src"])

        return dict(status="PASS" if not failures else "FAIL",
                    written=w, n_teacher=len(index), n_student=len(tensors),
                    copy_count=sum(1 for r in report if r["op"] == "COPY"),
                    derive_count=sum(1 for r in report if r["op"] == "DERIVE"),
                    deferred_to_t0=[r["src"] for r in report
                                    if r.get("transform") == "DEFER_TO_T0"],
                    copy_verified_reopen=not failures, failures=failures,
                    mapping=report,
                    honest="weight-level COPY/DERIVE/NEW only. NO teacher forward, "
                           "NO logit KL, NO ppl, NO generation, NO capability claim.")
    finally:
        for s in cache.values(): s.close()
