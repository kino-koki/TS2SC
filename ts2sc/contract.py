#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8 · Tokenizer/Embedding Binding Contract

This is the single gate that fixes the historical "只会吐乱码" failure.
V7 copied tokenizer files best-effort (missing files were silently OK), used a
self-made 2048/1024-token BPE prefix against a 151936-row embedding, and let
decode() fall back to raw byte-to-unicode strings. None of that is allowed here.

Layers:
  static_contract()  -- pure stdlib, runs on any machine (preflight)
  runtime_contract() -- needs transformers, runs at T+0 on the real release
"""
from __future__ import annotations

import json
import os

# Two real release shapes exist:
#   Qwen3-style  : six loose files (vocab.json + merges.txt separate)
#   MiniCPM5-style: tokenizer.json holds the BPE vocab/merges internally,
#                   chat template ships as a separate chat_template.jinja.
# Hardcoding either shape is exactly the R7-class bug v8 exists to kill.
TOKENIZER_FILES_REQUIRED = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "generation_config.json",
)
TOKENIZER_FILES_OPTIONAL = ("vocab.json", "merges.txt")
CHAT_TEMPLATE_FILES = ("chat_template.jinja",)
TOKENIZER_FILES = TOKENIZER_FILES_REQUIRED + TOKENIZER_FILES_OPTIONAL
# 133 B "version https://git-lfs.github.com/spec/v1" pointer: the exact
# historical failure recorded in TS2SC_VL47_PREFLIGHT.md and README_V7_PROD.md.
LFS_POINTER_MAX = 1024


def _is_lfs_pointer(p: str) -> bool:
    if not os.path.exists(p):
        return False
    b = open(p, "rb").read(256)
    return b.startswith(b"version https://git-lfs") or b.startswith(b"oid sha256:")


def static_contract(teacher_dir: str, student_dir: str, spec) -> dict:
    """Hard gate, no torch. Any FAIL -> refuse to emit a student checkpoint."""
    v = spec.values
    V = v.get("vocab_size")
    checks, failures = {}, []

    # 1. tokenizer files must be REAL, in the STUDENT dir, not best-effort.
    #    Optional files (vocab.json / merges.txt) are absent in modern BPE
    #    releases where tokenizer.json contains them -- absence is OK, but a
    #    present LFS pointer is never OK.
    for fn in TOKENIZER_FILES:
        required = fn in TOKENIZER_FILES_REQUIRED
        src = os.path.join(teacher_dir, fn)
        dst = os.path.join(student_dir, fn)
        present = os.path.exists(src)
        copied = os.path.exists(dst)
        pointer = _is_lfs_pointer(src) or _is_lfs_pointer(dst)
        checks["file_" + fn] = dict(teacher_present=present, copied=copied,
                                    required=required, lfs_pointer=pointer)
        if required and (not present or not copied):
            failures.append(f"TOKENIZER_FILE_MISSING:{fn}")
        if pointer:
            failures.append(f"LFS_POINTER_NOT_REAL_TOKENIZER:{fn}")

    # 2. id space must match the embedding matrix exactly.
    vj = os.path.join(student_dir, "vocab.json")
    n_vocab = None
    if os.path.exists(vj) and not _is_lfs_pointer(vj):
        try:
            obj = json.load(open(vj, encoding="utf-8"))
            n_vocab = len(obj) if isinstance(obj, dict) else None
        except Exception as e:
            failures.append(f"VOCAB_JSON_UNPARSEABLE:{type(e).__name__}")
    checks["vocab_rows"] = dict(config_vocab_size=V, vocab_json_rows=n_vocab)
    if V is None:
        failures.append("VOCAB_SIZE_UNDECLARED")
    elif n_vocab is not None and n_vocab != V:
        failures.append(f"VOCAB_EMBEDDING_MISMATCH:{n_vocab}!={V}")

    # 3. every special id must be inside the id space.
    sp = os.path.join(student_dir, "special_tokens_map.json")
    special = {}
    if os.path.exists(sp) and not _is_lfs_pointer(sp):
        try:
            special = json.load(open(sp, encoding="utf-8"))
        except Exception:
            pass
    ids = {}
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        val = special.get(key)
        if isinstance(val, dict):
            val = val.get("id")
        if isinstance(val, int):
            ids[key] = val
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if isinstance(v.get(key), int):
            ids[key] = v[key]
    checks["special_ids"] = ids
    for k, i in ids.items():
        if i is not None and V is not None and not (0 <= i < V):
            failures.append(f"SPECIAL_ID_OUT_OF_RANGE:{k}={i} V={V}")

    # 4. tied/untied must be INHERITED, not forced. This is the lm_head bug.
    tie = v.get("tie_word_embeddings")
    checks["tie_word_embeddings"] = dict(teacher=tie,
                                         inherited_by_student=spec.student_config().get("tie_word_embeddings"))
    if tie is None:
        failures.append("TIE_UNDECLARED")
    if tie is False:
        # teacher has a standalone lm_head -> student MUST carry it explicitly
        checks["lm_head_requirement"] = "COPY lm_head.weight (untied teacher)"
    else:
        checks["lm_head_requirement"] = "tied to embed_tokens; assert pointer identity at load"

    # 5. chat template must exist (native-level experience, not raw completion).
    tc = os.path.join(student_dir, "tokenizer_config.json")
    has_tpl, tpl_where = False, None
    if os.path.exists(tc) and not _is_lfs_pointer(tc):
        try:
            if "chat_template" in json.load(open(tc, encoding="utf-8")):
                has_tpl, tpl_where = True, "tokenizer_config.json"
        except Exception:
            pass
    if not has_tpl:
        for fn in CHAT_TEMPLATE_FILES:
            p = os.path.join(student_dir, fn)
            if os.path.exists(p) and not _is_lfs_pointer(p) and os.path.getsize(p) > 1024:
                has_tpl, tpl_where = True, fn
                break
    checks["chat_template_present"] = has_tpl
    checks["chat_template_source"] = tpl_where
    if not has_tpl:
        failures.append("CHAT_TEMPLATE_MISSING")

    return dict(passed=not failures, failures=failures, checks=checks,
                note="static only; runtime_contract() is mandatory at T+0")


def offline_tokenizer_check(tokenizer_json: str, vocab_json: str,
                            config: dict) -> dict:
    """v8.2 · Real tokenizer.json structure check WITHOUT transformers.

    runtime_contract() needs the transformers runtime (T+0). This offline
    twin inspects the raw JSON that ships with the release, so the R2/R3
    failure class can be caught before any GPU is involved.

    Checks: tokenizer.json parseable, model.vocab ids contiguous from 0,
    added_tokens ids inside vocab_size, max_id < embed rows, vocab.json
    row count consistent with config.vocab_size.
    """
    checks, failures = {}, []
    V = config.get("vocab_size")
    if V is None:
        return dict(passed=False, failures=["VOCAB_SIZE_UNDECLARED"])

    tj, vj = None, None
    try:
        tj = json.load(open(tokenizer_json, encoding="utf-8"))
        checks["tokenizer_json_parseable"] = True
    except Exception as e:
        failures.append("TOKENIZER_JSON_UNPARSEABLE:" + type(e).__name__)
    if vocab_json and os.path.exists(vocab_json):
        try:
            vj = json.load(open(vocab_json, encoding="utf-8"))
            checks["vocab_json_parseable"] = True
        except Exception as e:
            failures.append("VOCAB_JSON_UNPARSEABLE:" + type(e).__name__)
    else:
        # Many modern releases ship tokenizer.json only. The BPE model.vocab
        # inside it is the authoritative id space in that case.
        checks["vocab_json_parseable"] = "N/A (tokenizer.json only)"
        if tj:
            mv = tj.get("model", {}).get("vocab", {})
            if isinstance(mv, dict):
                checks["tokenizer_model_vocab_rows"] = len(mv)
                vj = mv

    if tj:
        vocab = tj.get("model", {}).get("vocab", {})
        ids = sorted(vocab.values()) if isinstance(vocab, dict) else []
        checks["model_vocab_entries"] = len(ids)
        if ids:
            checks["min_id"], checks["max_id"] = ids[0], ids[-1]
            if ids[0] != 0 or ids[-1] != len(ids) - 1:
                failures.append("MODEL_VOCAB_IDS_NOT_CONTIGUOUS_FROM_ZERO")
        added = tj.get("added_tokens", [])
        aids = [a.get("id") for a in added if isinstance(a, dict) and isinstance(a.get("id"), int)]
        checks["n_added_tokens"] = len(aids)
        checks["added_token_ids"] = [min(aids), max(aids)] if aids else []
        for i in aids:
            if not (0 <= i < V):
                failures.append(f"ADDED_TOKEN_ID_OUT_OF_RANGE:{i} V={V}")
                break
        all_ids = ids + aids
        if all_ids and max(all_ids) >= V:
            failures.append(f"MAX_TOKEN_ID_GE_VOCAB_SIZE:{max(all_ids)}>={V}")
        checks["max_id_lt_vocab_size"] = bool(all_ids and max(all_ids) < V)

    if isinstance(vj, dict):
        checks["vocab_rows"] = len(vj)
        # A standalone vocab.json must match embedding rows exactly. For
        # tokenizer.json-only releases the id space is model.vocab + added_tokens,
        # so the decisive test is max_id < vocab_size (checked above), NOT
        # len(model.vocab) == vocab_size (added tokens legitimately live above).
        if vocab_json and os.path.exists(vocab_json) and len(vj) != V:
            failures.append(f"VOCAB_JSON_ROWS_NEQ_CONFIG:{len(vj)}!={V}")

    return dict(passed=not failures, failures=failures, checks=checks,
                note="offline structural twin of runtime_contract(); "
                     "does NOT replace the real roundtrip test at T+0")


def runtime_contract(tokenizer, embed_shape, special_ids, chat_messages) -> dict:
    """Real tokenizer checks. Called only on the T+0 machine with transformers."""
    checks, failures = {}, []
    V = embed_shape[0]
    checks["embed_rows"] = V
    checks["len_tokenizer"] = len(tokenizer)
    if len(tokenizer) != V:
        failures.append(f"LEN_TOKENIZER_NEQ_EMBED_ROWS:{len(tokenizer)}!={V}")

    probes = [
        "华为公司是什么？",
        "请用一句话介绍你自己。",
        "Hello, how are you?",
        "中英混排 Mixed 123 😀，逗号、引号「」",
        "a" * 3 + "\u4e2d\u6587\u591a\u5b57\u8282\u8fb9\u754c" + "b" * 3,
    ]
    rts = []
    for s in probes:
        ids = tokenizer.encode(s, add_special_tokens=False)
        back = tokenizer.decode(ids, skip_special_tokens=True)
        over = [i for i in ids if not (0 <= i < V)]
        rts.append(dict(probe=s[:18], roundtrip_ok=(back == s), n_ids=len(ids),
                        out_of_range=len(over), decoded=back[:60]))
        if over:
            failures.append(f"ID_OUT_OF_RANGE_IN_PROBE:{len(over)}")
        if back != s:
            failures.append("ROUNDTRIP_MISMATCH:" + s[:12])
    checks["roundtrips"] = rts

    for k, i in special_ids.items():
        if i is not None and not (0 <= i < V):
            failures.append(f"SPECIAL_ID_OUT_OF_RANGE:{k}={i}")

    # UTF-8 streaming: decode the whole buffer, never one token at a time.
    s = probes[3]
    ids = tokenizer.encode(s, add_special_tokens=False)
    buf, streamed = [], ""
    for i in ids:
        buf.append(i)
        streamed = tokenizer.decode(buf, skip_special_tokens=True)
    checks["stream_utf8_boundary_ok"] = (streamed == s)
    if streamed != s:
        failures.append("STREAM_UTF8_BOUNDARY_BROKEN")

    try:
        tpl = tokenizer.apply_chat_template(chat_messages, tokenize=True,
                                            add_generation_prompt=True)
        checks["chat_template_ids"] = len(tpl)
        if not tpl:
            failures.append("CHAT_TEMPLATE_EMPTY")
    except Exception as e:
        failures.append(f"CHAT_TEMPLATE_FAILED:{type(e).__name__}")

    return dict(passed=not failures, failures=failures, checks=checks,
                note="G0 gate: any failure forbids claiming native-level behaviour")
