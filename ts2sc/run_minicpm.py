#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8.2 · Real MiniCPM5-1B pull / teacher verification / migration

Everything here runs against REAL bytes from kino-koki/MiniCPM5-1B.
No torch, no transformers, no safetensors package, no HuggingFace access.

Strategy forced by the sandbox (315 MiB disk, 3.47 GiB RAM, pypi/HF 403):
  * github git-blob API with `Accept: application/vnd.github.raw` streams the
    22 real byte shards into RAM (2.16 GB), never onto disk.
  * a virtual safetensors reader maps (abs_offset, length) -> shard slice.
  * pure-numpy Llama forward (RMSNorm / RoPE / GQA / SwiGLU) + real
    tokenizer.json BPE -> autoregressive generation.
  * weight-level migration is executed on REAL bytes; the full 2.16 GB
    student cannot be materialised on this disk, so we emit a probe
    checkpoint + a replayable recipe instead of faking a full model.

The GH token is read from the environment ONLY. It is never written to any
file, report, zip or log. Rotate/revoke after this run.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
import unicodedata
import urllib.request

import numpy as np

TOKEN = os.environ.get("GH_TOKEN", "")
REPO = "kino-koki/MiniCPM5-1B"
BASE = "/data/workspace/minicpm5_1b"
OUT = "/data/workspace/output"
PART = 99614720
LAST = 69381792
NPARTS = 22
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Real weight shards -> RAM
# ---------------------------------------------------------------------------
def gh(url, accept="application/vnd.github+json", timeout=240):
    h = {"User-Agent": "ts2sc-v8", "Accept": accept}
    if TOKEN:
        h["Authorization"] = "Bearer " + TOKEN
    r = urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)
    return r.read()


def shard_shas():
    d = json.loads(gh(f"https://api.github.com/repos/{REPO}/contents").decode())
    out = {}
    for x in d:
        m = re.fullmatch(r"model_part_(\d{2})", x["name"])
        if m:
            out[int(m.group(1))] = (x["sha"], x["size"])
    return out


class VirtualWeights:
    def __init__(self, part0: bytes, shas: dict, header: dict, ds: int):
        self.parts = {0: part0}
        self.shas = shas
        self.header = {k: v for k, v in header.items() if k != "__metadata__"}
        self.ds = ds
        self.n_fetched = 1
        self.bytes_fetched = len(part0)

    def part(self, i: int) -> bytes:
        if i in self.parts:
            return self.parts[i]
        sha, size = self.shas[i]
        last = None
        for attempt in range(4):
            try:
                raw = gh(f"https://api.github.com/repos/{REPO}/git/blobs/{sha}",
                         accept="application/vnd.github.raw", timeout=300)
                if len(raw) != size:
                    raise RuntimeError(f"part {i}: {len(raw)} != {size} (IncompleteRead)")
                self.parts[i] = raw
                self.n_fetched += 1
                self.bytes_fetched += len(raw)
                return raw
            except Exception as e:                  # transient CDN truncation
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"part {i} failed after 4 attempts: {type(last).__name__}: {last}")

    def preload(self):
        for i in range(NPARTS):
            self.part(i)
        return self.n_fetched, self.bytes_fetched

    def read(self, off: int, n: int) -> bytes:
        buf = bytearray()
        while n > 0:
            i = off // PART
            inner = off - i * PART
            p = self.part(i)
            k = min(n, len(p) - inner)
            if k <= 0:
                raise RuntimeError(f"read past shard at off={off} i={i}")
            buf += p[inner:inner + k]
            off += k
            n -= k
        return bytes(buf)

    def meta(self, name):
        return self.header[name]

    def raw(self, name) -> bytes:
        a, b = self.header[name]["data_offsets"]
        return self.read(self.ds + a, b - a)

    def rows(self, name, r0, r1) -> bytes:
        m = self.header[name]
        shape = m["shape"]
        assert len(shape) == 2, name
        itemsize = 2 if m["dtype"] == "BF16" else 4
        a, _ = m["data_offsets"]
        start = self.ds + a + r0 * shape[1] * itemsize
        return self.read(start, (r1 - r0) * shape[1] * itemsize)

    def f32(self, name) -> np.ndarray:
        m = self.header[name]
        assert m["dtype"] == "BF16", name
        u = np.frombuffer(self.raw(name), dtype=np.uint16)
        a = (u.astype(np.uint32) << 16).view(np.float32)
        return a.reshape(m["shape"])

    def rows_f32(self, name, r0, r1) -> np.ndarray:
        m = self.header[name]
        u = np.frombuffer(self.rows(name, r0, r1), dtype=np.uint16)
        a = (u.astype(np.uint32) << 16).view(np.float32)
        return a.reshape(r1 - r0, m["shape"][1])


def bf16_to_f32(b: bytes) -> np.ndarray:
    u = np.frombuffer(b, dtype=np.uint16)
    return (u.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------------------
# 2. Real tokenizer.json -> pure-stdlib BPE codec
# ---------------------------------------------------------------------------
def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}


class MiniCPMTokenizer:
    def __init__(self, path):
        t = json.load(open(path, encoding="utf-8"))
        self.vocab = t["model"]["vocab"]                 # token str -> id
        self.id2tok = {v: k for k, v in self.vocab.items()}
        merges = t["model"]["merges"]
        self.rank = {}
        for i, m in enumerate(merges):
            if isinstance(m, str):
                a, b = m.split(" ", 1)
            else:
                a, b = m[0], m[1]
            self.rank[(a, b)] = i
        self.added = {a["content"]: (a["id"], a.get("special", False))
                      for a in t.get("added_tokens", [])}
        self.special_ids = {i for i, sp in self.added.values() if sp}
        self.n_vocab = len(self.vocab)
        self.n_merges = len(self.rank)
        self.max_id = max(self.vocab.values())

    @staticmethod
    def _pretokenize(text):
        """Reversible approximation of HF's ByteLevel Sequence pre-tokenizer.

        Python `re` lacks \\p{L}; we use a Unicode-aware scanner whose chunks
        always concatenate back to the original string (checked by roundtrip).
        """
        out, buf, kind = [], "", None
        def flush():
            nonlocal buf, kind
            if buf:
                out.append(buf); buf = ""; kind = None
        for ch in text:
            cat = unicodedata.category(ch)
            if ch.isspace() or ch in "\r\n":
                k = "S"
            elif cat[0] == "L" or ch == "_":
                k = "L"
            elif cat[0] == "N":
                k = "N"
            else:
                k = "O"
            if kind is None:
                kind = k; buf = ch
            elif k == kind:
                buf += ch
            else:
                flush(); kind = k; buf = ch
        flush()
        # split long digit runs into <=3 (HF \p{N}{1,3})
        out2 = []
        for c in out:
            if c and unicodedata.category(c[0])[0] == "N" and len(c) > 3:
                out2 += [c[i:i + 3] for i in range(0, len(c), 3)]
            else:
                out2.append(c)
        assert "".join(out2) == text, "pretokenizer not reversible"
        return out2

    def _bpe(self, chars):
        w = list(chars)
        while len(w) > 1:
            best, bi = None, None
            for i in range(len(w) - 1):
                r = self.rank.get((w[i], w[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi is None:
                break
            w[bi:bi + 2] = [w[bi] + w[bi + 1]]
        return w

    def encode(self, text, add_bos=False):
        ids = [0] if add_bos else []          # <s> via post_processor
        # special / added tokens are matched verbatim, longest first
        specials = sorted((c for c in self.added), key=len, reverse=True)
        parts, rest = [], text
        while rest:
            hit = next((c for c in specials if rest.startswith(c)), None)
            if hit:
                parts.append(("SPECIAL", hit)); rest = rest[len(hit):]
            else:
                # accumulate until next special
                j = len(rest)
                for c in specials:
                    p = rest.find(c)
                    if 0 <= p < j: j = p
                parts.append(("TEXT", rest[:j])); rest = rest[j:]
        unknown = []
        for typ, s in parts:
            if typ == "SPECIAL":
                ids.append(self.added[s][0]); continue
            for chunk in self._pretokenize(s):
                chars = "".join(B2U[b] for b in chunk.encode("utf-8"))
                for piece in self._bpe(chars):
                    if piece in self.vocab:
                        ids.append(self.vocab[piece])
                    elif len(piece) == 1 and piece in U2B:
                        ids.append(self.vocab.get(piece, 3))
                    else:
                        for c in piece:
                            if c in self.vocab: ids.append(self.vocab[c])
                            else: unknown.append(c); ids.append(3)
        return ids, unknown

    def decode(self, ids, skip_special=True):
        s = ""
        for i in ids:
            i = int(i)
            if skip_special and i in self.special_ids:
                continue
            if i in self.id2tok:
                s += self.id2tok[i]
            elif i in self.added_by_id:
                s += self.added_by_id[i]
            else:
                s += ""
        raw = bytearray()
        for c in s:
            raw.append(U2B.get(c, ord(c) & 0xFF))
        return bytes(raw).decode("utf-8", errors="replace")

    @property
    def added_by_id(self):
        return {i: c for c, (i, _) in self.added.items()}


# ---------------------------------------------------------------------------
# 3. Pure-numpy Llama teacher
# ---------------------------------------------------------------------------
class TeacherLM:
    def __init__(self, V: VirtualWeights, cfg: dict):
        self.V = V
        self.d = cfg["hidden_size"]
        self.nl = cfg["num_hidden_layers"]
        self.nq = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.h = cfg["intermediate_size"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_theta"]
        self.vocab = cfg["vocab_size"]
        self.norm_w = V.f32("model.norm.weight")
        self.rep = self.nq // self.nkv
        inv = 1.0 / (self.theta ** (np.arange(0, self.hd, 2, dtype=np.float64) / self.hd))
        self.inv = inv.astype(np.float32)

    @staticmethod
    def rms(x, w, eps):
        return x * (1.0 / np.sqrt(np.mean(x * x) + eps)) * w

    def rope(self, x, pos):
        # x: (heads, hd) or (T, heads, hd); rotate-half (HF Llama)
        f = np.outer(np.array([pos], dtype=np.float64), self.inv.astype(np.float64))
        emb = np.concatenate([f, f], axis=-1).astype(np.float32)[0]   # (hd,)
        cos, sin = np.cos(emb), np.sin(emb)
        def rh(a):
            a1, a2 = a[..., :self.hd // 2], a[..., self.hd // 2:]
            return np.concatenate([-a2, a1], axis=-1)
        return x * cos + rh(x) * sin

    def layer(self, i):
        p = f"model.layers.{i}."
        W = {}
        W["in"] = self.V.f32(p + "input_layernorm.weight")
        W["q"] = self.V.f32(p + "self_attn.q_proj.weight")
        W["k"] = self.V.f32(p + "self_attn.k_proj.weight")
        W["v"] = self.V.f32(p + "self_attn.v_proj.weight")
        W["o"] = self.V.f32(p + "self_attn.o_proj.weight")
        W["post"] = self.V.f32(p + "post_attention_layernorm.weight")
        W["gate"] = self.V.f32(p + "mlp.gate_proj.weight")
        W["up"] = self.V.f32(p + "mlp.up_proj.weight")
        W["down"] = self.V.f32(p + "mlp.down_proj.weight")
        return W

    def step(self, tid, pos, cache, need_logits=True, fusion_hook=None,
             return_hidden=False, logit_bias_fn=None):
        x = self.V.rows_f32("model.embed_tokens.weight", tid, tid + 1)[0]
        for l in range(self.nl):
            W = self.layer(l)
            h = self.rms(x, W["in"], self.eps)
            q = (W["q"] @ h).reshape(self.nq, self.hd)
            k = (W["k"] @ h).reshape(self.nkv, self.hd)
            v = (W["v"] @ h).reshape(self.nkv, self.hd)
            q = self.rope(q, pos)
            k = self.rope(k, pos)
            K, Vc = cache[l]
            K = np.concatenate([K, k[None]], axis=0)
            Vc = np.concatenate([Vc, v[None]], axis=0)
            cache[l] = (K, Vc)
            kr = np.repeat(K, self.rep, axis=1)     # (T, nq, hd)
            vr = np.repeat(Vc, self.rep, axis=1)
            s = np.einsum("hd,thd->ht", q, kr) / math.sqrt(self.hd)
            s = s - s.max(axis=-1, keepdims=True)
            p = np.exp(s); p /= p.sum(axis=-1, keepdims=True)
            o = np.einsum("ht,thd->hd", p, vr).reshape(self.nq * self.hd)
            x = x + W["o"] @ o
            h2 = self.rms(x, W["post"], self.eps)
            g = W["gate"] @ h2
            u = W["up"] @ h2
            ff = W["down"] @ (g / (1.0 + np.exp(-g)) * u)
            x = x + ff
            if fusion_hook is not None:
                x = x + fusion_hook(l, x, pos)
            del W, h, q, k, v, K, Vc, kr, vr, s, p, o, g, u, ff
        x = self.rms(x, self.norm_w, self.eps)
        if return_hidden:
            return x.copy()                    # pre-lm_head hidden, (d,)
        if not need_logits:
            return None
        bias = logit_bias_fn(x) if logit_bias_fn is not None else None
        if isinstance(bias, tuple):
            return self.logits(x, bias=bias[0], relative_margin=bias[1])
        return self.logits(x, bias=bias)

    def logits(self, x, topk=8, bias=None, relative_margin=None):
        # T3' (v8.4): the OLD initial sentinel -1 survived all-NaN logits
        # because `NaN > best_v` is always False, silently producing 24
        # out-of-range ids and an empty decode. Use a LEGAL initial id and
        # expose a finite flag so the decode gate can reject non-finite runs.
        #
        # v8.8: optional OUTPUT-LAYER adapter. `bias` maps token_id -> weight
        # in (0,1]. If `relative_margin` is set, only those |S| logits are
        # raised to (current max + margin*weight); every other logit is left
        # byte-identical, so teacher decoding is untouched everywhere else.
        # This is NOT a hidden-state perturbation: injecting before final
        # RMSNorm (v8.7) was re-normalised away and corrupted semantics.
        chunks = []
        for r0 in range(0, self.vocab, 8192):
            r1 = min(r0 + 8192, self.vocab)
            Wb = self.V.rows_f32("lm_head.weight", r0, r1)
            chunks.append(Wb @ x)
        all_lg = np.concatenate(chunks)
        finite_all = bool(np.isfinite(all_lg).all())
        if bias and relative_margin is not None:
            # v9.0 覆盖模式：把目标 token 钉到当前 top-1 之上。
            # 实测在多步引导下会把同一个 token 反复推到 top1 → 重复退化
            # （"青小小小小…" / "Tuan Tuan Tuan…"），v9.1 起改用加法模式。
            m = float(all_lg[np.isfinite(all_lg)].max()) if finite_all else \
                float(np.nanmax(all_lg))
            for tid, w in bias.items():
                if 0 <= int(tid) < self.vocab and w > 0:
                    all_lg[int(tid)] = max(float(all_lg[int(tid)]),
                                           m + float(relative_margin) * float(w))
        elif bias:
            # v9.1 additive：只做加法，teacher 原始 logit 仍参与竞争，
            # 身份 token 被「推一把」而不是「钉死」⇒ 不会锁成复读机。
            for tid, w in bias.items():
                t = int(tid)
                if 0 <= t < self.vocab:
                    all_lg[t] += float(w)
        tops = []
        if all_lg.size >= topk and finite_all:
            ix = np.argpartition(all_lg, -topk)[-topk:]
            for t in ix:
                tops.append((float(all_lg[t]), int(t)))
        tops.sort(reverse=True)
        best_i = int(np.argmax(all_lg)) if finite_all else 0
        best_v = float(all_lg[best_i]) if finite_all else -1e30
        self.last_logits_finite = finite_all
        return best_i, [i for _, i in tops[:topk]], best_v

    def generate(self, prompt_ids, max_new=28):
        """Greedy autoregression. step(t) adds t's K/V and returns the
        distribution for token t+1. Hence the LAST prompt token yields the
        first generated token, and each generated token yields the next."""
        cache = [(np.zeros((0, self.nkv, self.hd), np.float32),
                  np.zeros((0, self.nkv, self.hd), np.float32)) for _ in range(self.nl)]
        t0 = time.time()
        L = len(prompt_ids)
        r = None
        for pos, tid in enumerate(prompt_ids):
            r = self.step(int(tid), pos, cache, need_logits=(pos == L - 1))
        if r is None:
            return [], dict(elapsed_s=0, n_prompt=L, n_new=0)
        out = [r[0]]
        for j in range(1, max_new):
            r = self.step(int(out[-1]), L + j - 1, cache, need_logits=True)
            out.append(r[0])
            if r[0] in (1, 130073):
                break
        return out, dict(elapsed_s=round(time.time() - t0, 1),
                         n_prompt=L, n_new=len(out))


# ---------------------------------------------------------------------------
# 4. Real byte-level migration probe + replay recipe
# ---------------------------------------------------------------------------
def write_safetensors(path, tensors, metadata=None):
    hdr = {}
    off, blobs = 0, []
    for name, (dtype, shape, raw) in tensors.items():
        n = 1
        for x in shape: n *= x
        if len(raw) != n * (2 if dtype == "BF16" else 4):
            raise ValueError(f"{name}: byte/shape mismatch")
        hdr[name] = dict(dtype=dtype, shape=list(shape), data_offsets=[off, off + len(raw)])
        blobs.append(raw); off += len(raw)
    if metadata:
        hdr["__metadata__"] = metadata
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((-len(hb)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for b in blobs: f.write(b)
    return dict(path=path, n_tensors=len(tensors), data_bytes=off)


def migration_probe(V, out_dir, cfg):
    """Real COPY + real GQA row-remap DERIVE on teacher bytes."""
    os.makedirs(out_dir, exist_ok=True)
    d = cfg["hidden_size"]; kv_dim = cfg["num_key_value_heads"] * cfg["head_dim"]
    R = d // kv_dim
    sys.path.insert(0, "/data/workspace/ts2sc_v8")
    from modeling_v8 import build_slot_map
    slot = build_slot_map(d, kv_dim, cfg["num_attention_heads"])
    tensors, checks = {}, []
    p = "model.layers.0."
    # COPY: layer-0 norms + FFN + o_proj, verbatim bytes
    copy_names = [p + "input_layernorm.weight", p + "post_attention_layernorm.weight",
                  p + "mlp.down_proj.weight", p + "mlp.gate_proj.weight",
                  p + "mlp.up_proj.weight", p + "self_attn.o_proj.weight"]
    for n in copy_names:
        raw = V.raw(n); tensors[n] = (V.meta(n)["dtype"], V.meta(n)["shape"], raw)
        checks.append(dict(op="COPY", tensor=n, dst=n, bytes=len(raw),
                           sha256=hashlib.sha256(raw).hexdigest()[:16],
                           bit_identical=True))
    # DERIVE: k/v/o/q projections. No zero padding: repeat REAL teacher rows.
    for n in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):
        full = p + n
        m = V.meta(full); raw = V.raw(full); shape = m["shape"]  # (kv_dim, d)
        rowlen = shape[1] * 2
        out = b"".join(raw[j * rowlen:(j + 1) * rowlen] for j in slot["row_to_teacher"])
        dst = "sgc." + n
        tensors[dst] = ("BF16", [d, d], out)
        checks.append(dict(op="DERIVE", tensor=full, dst=dst, transform="GQA_ROW_REMAP",
                           in_shape=shape, out_shape=[d, d],
                           zero_padded_rows=0, all_rows_mapped=True))
    # q_proj COPY verbatim (no reshape needed)
    for n in ("self_attn.q_proj.weight",):
        full = p + n; raw = V.raw(full); dst = "sgc." + n
        tensors[dst] = (V.meta(full)["dtype"], V.meta(full)["shape"], raw)
        checks.append(dict(op="COPY", tensor=full, dst=dst, bytes=len(raw), bit_identical=True))
    # NEW: zero-init branch params (F32), shape from real config
    for k, shp in (("Wc", [3, d]), ("Wg.weight", [d, d]), ("wgate.weight", [1, d]),
                   ("alpha", [1]), ("beta", [1])):
        tensors["new." + k] = ("F32", shp, b"\x00" * (int(np.prod(shp)) * 4))
    w = write_safetensors(os.path.join(out_dir, "student_probe.safetensors"), tensors,
                          metadata=dict(format="pt", ts2sc="v8.2", scope="PROBE_ONLY"))
    # reopen and verify COPY bytes from the WRITTEN file
    from safetensors_v8 import SafeTensors
    with SafeTensors(os.path.join(out_dir, "student_probe.safetensors")) as s:
        for c in checks:
            if c["op"] != "COPY": continue
            key = c["dst"]
            c["reopened_key_present"] = key in s.keys
            c["bit_identical"] = bool(key in s.keys and s.tensor(key) == V.raw(c["tensor"]))
    # DERIVE: verify written tensor is (d,d), all rows are REAL teacher rows
    for c in checks:
        if c["op"] == "DERIVE":
            with SafeTensors(os.path.join(out_dir, "student_probe.safetensors")) as s:
                b = s.tensor(c["dst"])
                src = V.raw(c["tensor"])
                rowlen = c["in_shape"][1] * 2
                c["written_bytes_ok"] = (len(b) == d * d * 2)
                c["no_zero_rows"] = bool(b) and all(
                    b[i * rowlen:(i + 1) * rowlen] != b"\x00" * rowlen for i in range(d))
    student_cfg = dict(cfg)
    student_cfg.update(model_type="simplcore_v8", architectures=["SimplCoreV8ForCausalLM"],
                       n_slots=R, conv_kernel=3, rho=0.90,
                       fusion="PARALLEL_GATED", beta_init=0.0,
                       note="teacher attention / FFN / norms / lm_head KEPT verbatim")
    json.dump(student_cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    json.dump(dict(checks=checks, slot=dict(n_slots=R, kv_dim=kv_dim,
                                            zero_padded_rows=slot["zero_padded_rows"]),
                   written=w),
              open(os.path.join(out_dir, "migration_probe.json"), "w"), indent=2)
    return dict(written=w, n_checks=len(checks), n_copy=sum(1 for c in checks if c["op"] == "COPY"),
                n_derive=sum(1 for c in checks if c["op"] == "DERIVE"),
                all_copy_bit_identical=all(c.get("bit_identical", True) for c in checks),
                zero_padded_rows=slot["zero_padded_rows"], n_slots=R)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    rep = dict(title="TS2SC v8.2 · MiniCPM5-1B real pull / teacher verify / migration",
               date=time.strftime("%Y-%m-%d %H:%M:%S"),
               repo=REPO, honest="No torch, no transformers, no HF access, no token in any file.")

    # -- A. real files already on disk
    cfg = json.load(open(os.path.join(BASE, "config.json"), encoding="utf-8"))
    idx = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))
    rep["A_real_config"] = dict(hidden_size=cfg["hidden_size"], layers=cfg["num_hidden_layers"],
                                vocab_size=cfg["vocab_size"], nq=cfg["num_attention_heads"],
                                nkv=cfg["num_key_value_heads"], head_dim=cfg["head_dim"],
                                tie_word_embeddings=cfg["tie_word_embeddings"],
                                rope_theta=cfg["rope_theta"], dtype=cfg["torch_dtype"])
    rep["A_index"] = dict(n_tensors=len(idx["weight_map"]),
                          total_size=idx["metadata"]["total_size"])

    # -- B. tokenizer codec on REAL 9.9 MB tokenizer.json
    tok = MiniCPMTokenizer(os.path.join(BASE, "tokenizer.json"))
    probes = ["华为公司是什么？", "请用一句话介绍你自己。", "Hello, how are you?",
              "中英混排 Mixed 123 😀，逗号、引号「」", "a中文多字节边界b"]
    rt = []
    for s in probes:
        ids, unk = tok.encode(s, add_bos=True)
        back = tok.decode(ids, skip_special=True)
        over = [i for i in ids if not (0 <= i < cfg["vocab_size"])]
        rt.append(dict(probe=s, n_ids=len(ids), roundtrip_ok=(back == s),
                       decoded=back, out_of_range=len(over), unknown_chars=len(unk)))
    rep["B_tokenizer"] = dict(n_vocab=tok.n_vocab, n_merges=tok.n_merges,
                              max_vocab_id=tok.max_id, config_vocab_size=cfg["vocab_size"],
                              max_id_lt_vocab_size=tok.max_id < cfg["vocab_size"],
                              decoder="ByteLevel", roundtrips=rt,
                              all_roundtrip_ok=all(r["roundtrip_ok"] for r in rt),
                              all_ids_in_range=all(r["out_of_range"] == 0 for r in rt))
    print("[B] tokenizer roundtrips:", sum(r["roundtrip_ok"] for r in rt), "/", len(rt), flush=True)
    for r in rt:
        print("   ", "OK " if r["roundtrip_ok"] else "FAIL", repr(r["probe"][:20]),
              "->", repr(r["decoded"][:40]), flush=True)

    # -- C. real weight bytes
    part0 = open(os.path.join(BASE, "model_part_00"), "rb").read()
    n = struct.unpack("<Q", part0[:8])[0]
    hdr = json.loads(part0[8:8 + n].decode())
    ds = 8 + n
    shas = shard_shas()
    V = VirtualWeights(part0, shas, hdr, ds)
    nf, nb = V.preload()
    T = V.header
    emb = V.rows_f32("model.embed_tokens.weight", 0, 256)
    rep["C_weights"] = dict(n_parts=nf, bytes_in_ram=nb,
                            n_tensors=len(T), header_len=n, data_start=ds,
                            dtypes=sorted({v["dtype"] for v in T.values()}),
                            embed_prefix=dict(n_values=emb.size, finite=bool(np.isfinite(emb).all()),
                                              mean=float(emb.mean()), std=float(emb.std()),
                                              zero_frac=float((emb == 0).mean())),
                            lm_head_present="lm_head.weight" in T,
                            tie_inherited=cfg["tie_word_embeddings"],
                            real_trained_weights=bool(np.isfinite(emb).all() and emb.std() > 1e-4))
    print(f"[C] {nf} real shards / {nb/2**30:.2f} GiB in RAM, {len(T)} tensors, "
          f"embed std={emb.std():.5f}", flush=True)

    # -- D. TEACHER generation (the "会说话" verification)
    lm = TeacherLM(V, cfg)
    prompts = [
        ("raw", "请用一句话介绍你自己。"),
        ("chatml", "<s><|im_start|>user\n请用一句话介绍你自己。<|im_end|>\n<|im_start|>assistant\n"),
        ("zh_q", "中国的首都是"),
        ("en", "Hello, how are you?"),
    ]
    gens = []
    for tag, text in prompts:
        ids, unk = tok.encode(text, add_bos=(tag == "raw"))
        try:
            out, meta = lm.generate(ids, max_new=24)
            dec = tok.decode(out, skip_special=True)
            # readability proxies (NOT capability claims)
            cjk = sum(1 for c in dec if "\u4e00" <= c <= "\u9fff")
            bad = sum(1 for c in dec if 0x00A0 <= ord(c) <= 0x024F)
            gens.append(dict(tag=tag, prompt=text[:40], n_prompt=len(ids),
                             n_new=len(out), generated=dec,
                             cjk_chars=cjk, latin_mojibake_chars=bad,
                             readable_proxy=bool(len(dec) > 8 and bad <= 2 and
                                                 (cjk > 0 or re.search(r"[A-Za-z]{3}", dec))),
                             elapsed_s=meta["elapsed_s"]))
            print(f"[D] {tag:7s} {meta['elapsed_s']:6.1f}s -> {dec[:90]!r}", flush=True)
        except Exception as e:
            gens.append(dict(tag=tag, error=type(e).__name__ + ":" + str(e)[:120]))
            print(f"[D] {tag} ERROR {e}", flush=True)
    rep["D_teacher_generation"] = dict(
        engine="pure-numpy Llama forward on REAL BF16 bytes (no torch)",
        constraint="2 CPU / 3.47 GiB RAM; weights streamed into RAM, not on disk",
        generations=gens,
        any_readable=any(g.get("readable_proxy") for g in gens),
        caveat="This is a self-written forward. A negative result is an "
               "implementation/environment outcome, NOT evidence that the teacher "
               "cannot speak. Only the real HF/transformers run at T+0 is decisive.")
    rep["D_teacher_generation"]["verified_teacher_speaks"] = bool(
        rep["D_teacher_generation"]["any_readable"])

    # -- E. real byte-level migration probe
    mp = migration_probe(V, os.path.join(OUT, "minicpm5_student_probe"), cfg)
    rep["E_migration_probe"] = mp
    print(f"[E] migration probe: {mp['n_copy']} COPY / {mp['n_derive']} DERIVE / "
          f"copy_ok={mp['all_copy_bit_identical']} zero_pad={mp['zero_padded_rows']}", flush=True)

    rep["total_elapsed_s"] = round(time.time() - t0, 1)
    rep["boundaries"] = [
        "Full 2.16 GB student NOT written to disk (315 MiB free).",
        "No teacher forward via transformers; no logit KL, no ppl, no paired CI.",
        "No repository entries were deleted or modified.",
        "No push was performed.",
        "GH token used from environment only; revoke it now.",
    ]
    p = os.path.join(OUT, "minicpm5_teacher_verify.json")
    json.dump(rep, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n[done]", p, "total", rep["total_elapsed_s"], "s", flush=True)


if __name__ == "__main__":
    main()
