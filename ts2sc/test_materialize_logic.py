#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 materialize_full_student_v10_3.py 的组装逻辑真能跑通。

不下载 2 GiB：构造 5 张量的 mini 版本，按脚本硬编码的文件名摆放，
真实执行脚本，检查输出 safetensors 的 5 条逐条 sha256 全部匹配。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import push_full_weights as W  # noqa: E402

ROOT = "/data/workspace/output/materialize_test"
RESULT = "/data/workspace/output/materialize_logic_test.json"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def make_st(tensors, meta=None):
    """tensors: name -> (dtype, shape, raw bytes)"""
    hdr, off, blobs = {}, 0, []
    for name, (dt, shape, raw) in tensors.items():
        hdr[name] = dict(dtype=dt, shape=list(shape),
                         data_offsets=[off, off + len(raw)])
        blobs.append(raw)
        off += len(raw)
    hdr["__metadata__"] = meta or dict(format="pt")
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    return struct.pack("<Q", len(hb)) + hb + b"".join(blobs), hdr


def main():
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
    os.makedirs(os.path.join(ROOT, "teacher_shards"))
    os.makedirs(os.path.join(ROOT, "weights_v10"))

    rng = np.random.default_rng(20261128)
    # teacher: 2 个 BF16 COPY 张量
    t1 = rng.integers(0, 65535, 8, dtype=np.uint16).tobytes()      # [2,2] BF16
    t2 = rng.integers(0, 65535, 12, dtype=np.uint16).tobytes()     # [3,2] BF16
    teacher, thdr = make_st({
        "model.embed_tokens.weight": ("BF16", [2, 2], t1),
        "model.norm.weight": ("BF16", [3, 2], t2),
    })
    open(os.path.join(ROOT, "teacher_shards/part_00_00.bin"), "wb").write(
        teacher)
    open(os.path.join(ROOT, "teacher_shards/MANIFEST.json"), "w",
         encoding="utf-8").write(json.dumps(dict(
             schema="TS2SC/v10.3-full-teacher-weights",
             n_upstream_shards=22, n_subshards=1,
             subshard_bytes=W.CHUNK,
             subshards=[dict(path="teacher_shards/part_00_00.bin",
                             upstream_shard=0, offset=0,
                             bytes=len(teacher), sha256=sha(teacher))],
             total_bytes=len(teacher), teacher_sha256=sha(teacher)), indent=2))

    # branch: 2 个 SGC NEW (F32)
    n1 = rng.normal(0, 1, 6).astype(np.float32).tobytes()
    n2 = rng.normal(0, 1, 4).astype(np.float32).tobytes()
    branch, bhdr = make_st({
        "model.blocks.0.sgc_v10.core.Win": ("F32", [2, 3], n1),
        "model.blocks.0.sgc_v10.Wdown": ("F32", [2, 2], n2),
    })
    open(os.path.join(ROOT, "weights_v10/student_branch_v10.part00"),
         "wb").write(branch)

    # SMM: 1 个 NEW
    s1 = rng.normal(0, 1, 4).astype(np.float32).tobytes()
    smm, shdr = make_st({
        "model.blocks.0.smm.W_c": ("F32", [2, 2], s1),
    })
    open(os.path.join(ROOT, "student_smm_v10_3.safetensors"), "wb").write(smm)

    # scoped manifest: 5 条，逐条 sha256 与真实字节一致
    entries = []
    for name, dt, shape, raw, op in [
        ("model.embed_tokens.weight", "BF16", [2, 2], t1, "COPY"),
        ("model.norm.weight", "BF16", [3, 2], t2, "COPY"),
        ("model.blocks.0.sgc_v10.core.Win", "F32", [2, 3], n1, "NEW"),
        ("model.blocks.0.sgc_v10.Wdown", "F32", [2, 2], n2, "NEW"),
        ("model.blocks.0.smm.W_c", "F32", [2, 2], s1, "NEW"),
    ]:
        entries.append(dict(name=name, op=op, dtype=dt, shape=shape,
                            bytes=len(raw), sha256=sha(raw)))
    open(os.path.join(ROOT, "student_scoped_manifest_v10_3.json"), "w",
         encoding="utf-8").write(json.dumps(dict(
             schema="TS2SC/v10.3-scoped-manifest", n_tensors=5,
             counts=dict(COPY=2, DERIVE=0, NEW=3), unclassified=0,
             entries=entries), indent=2))

    open(os.path.join(ROOT, "materialize_full_student_v10_3.py"), "w",
         encoding="utf-8").write(W.MATERIALIZE)

    r = subprocess.run(
        [sys.executable, "materialize_full_student_v10_3.py",
         "--out", "student_full.safetensors"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr

    ok, detail = False, {}
    if r.returncode == 0:
        b = open(os.path.join(ROOT, "student_full.safetensors"), "rb").read()
        n = struct.unpack("<Q", b[:8])[0]
        hdr = json.loads(b[8:8 + n])
        ds = 8 + n
        bad = []
        for e in entries:
            a, z = hdr[e["name"]]["data_offsets"]
            raw = b[ds + a: ds + z]
            if sha(raw) != e["sha256"]:
                bad.append(e["name"])
        detail = dict(n_tensors_in_output=len([k for k in hdr
                                               if k != "__metadata__"]),
                      all_sha256_verified=bool(len(bad) == 0),
                      mismatches=bad,
                      file_bytes=len(b),
                      header_metadata=hdr["__metadata__"])
        ok = bool(len(bad) == 0 and detail["n_tensors_in_output"] == 5)

    rep = dict(
        title="materialize_full_student_v10_3.py 组装逻辑小规模真跑",
        method="5 mini tensors (2 COPY + 3 NEW), real script execution, "
               "output safetensors re-opened and sha256 re-checked",
        returncode=r.returncode, stdout=out.strip()[-600:],
        detail=detail, passed=ok,
        honest="This validates ASSEMBLY LOGIC only, not the 2.01 GiB "
               "full-scale materialisation (disk cannot hold it here).")
    json.dump(rep, open(RESULT, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return rep


if __name__ == "__main__":
    main()
