#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TS2SC v8.2 · Credential hygiene gate

A token was pasted into the conversation this round. The project's own
TS2SC_VL47_PREFLIGHT.md §7 already records the same incident ("token appeared
in plaintext multiple times ... revoke"). So this is not a hypothetical:
credentials must NEVER enter a deliverable, a zip, a report, a state file or
a git commit.

Design rules:
  * matched VALUES are never printed, logged or written to any file;
    only file:line + pattern name + REDACTED is reported.
  * the scanner's own source is safe to scan: patterns are expressed as
    rule objects, not literal secrets.
  * scan() is called by the packer and by preflight part_D. Any hit -> refuse.
"""
from __future__ import annotations

import os
import re

# (name, compiled regex). Built from fragments so no literal credential exists.
RULES = [
    ("github_classic_pat", re.compile(r"gh[pousr]_" + r"[A-Za-z0-9]{36}")),
    ("github_fine_grained_pat", re.compile(r"github_pat_" + r"[A-Za-z0-9_]{50,}")),
    ("openai_key", re.compile(r"sk-" + r"[A-Za-z0-9_\-]{20,}")),
    ("aws_access_key", re.compile(r"AKIA" + r"[0-9A-Z]{16}")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer_in_source", re.compile(r"(?i)authorization\s*[:=]\s*(?:bearer|token)\s+\S{16,}")),
    ("env_secret_assignment", re.compile(r"(?i)(?:GH_TOKEN|GITHUB_TOKEN|HF_TOKEN|API_KEY)\s*=\s*['\"]?[A-Za-z0-9_\-]{16,}")),
    ("lfs_oid_pointer", re.compile(r"oid sha256:[0-9a-f]{64}")),
]
SCAN_EXT = {".py", ".md", ".json", ".txt", ".cfg", ".ini", ".yaml", ".yml",
            ".sh", ".ps1", ".toml", ".tex", ".bib", ".csv", ".zip"}
REDACTED = "[REDACTED]"


def scan_text(text: str, label: str = "<text>") -> list:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        for name, rx in RULES:
            if rx.search(line):
                hits.append(dict(file=label, line=i, rule=name, match=REDACTED))
    return hits


def scan_paths(paths: list, follow_dirs: bool = True) -> dict:
    files = []
    for p in paths:
        if os.path.isdir(p) and follow_dirs:
            for dp, dn, fn in os.walk(p):
                dn[:] = [d for d in dn if d not in (".git", "__pycache__", "_selftest", "_run", "_test")]
                for f in fn:
                    if os.path.splitext(f)[1].lower() in SCAN_EXT:
                        files.append(os.path.join(dp, f))
        elif os.path.isfile(p):
            files.append(p)
    hits, scanned = [], []
    for f in files:
        if f.endswith(".zip"):
            continue                      # zips are binary; members scanned by packer
        try:
            txt = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        scanned.append(f)
        hits += scan_text(txt, os.path.relpath(f, "/data/workspace"))
    return dict(passed=not hits, n_files=len(scanned), n_hits=len(hits),
                hits=hits, note="values are redacted; no secret is ever written")


def assert_clean(paths: list):
    r = scan_paths(paths)
    if not r["passed"]:
        raise SystemExit("[secrets] REFUSING TO PACKAGE: "
                         + str(r["n_hits"]) + " credential-like hits. "
                         + "Rotate/revoke and remove them before retrying.")
    return r


if __name__ == "__main__":
    import json, sys
    print(json.dumps(scan_paths(sys.argv[1:] or ["."]), indent=2, ensure_ascii=False))
