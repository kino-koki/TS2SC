#!/usr/bin/env python3
"""拉取 MiniCPM5-1B 分片 → 合并为 model.safetensors（边下边拼，节空间）"""
import json, os, sys, time, urllib.request
from pathlib import Path

TOKEN = os.environ["GH_TOKEN"]
REPO = os.getenv("MC_REPO", "kino-koki/MiniCPM5-1B")
OUT = Path("/data/workspace/mcpm5"); OUT.mkdir(exist_ok=True)
API = "https://api.github.com"
H = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
     "User-Agent": "t/1"}


def api(path):
    r = urllib.request.Request(API + path, headers=H)
    return json.loads(urllib.request.urlopen(r, timeout=30).read())


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open("/data/workspace/dl_mcpm.log", "a") as f:
        f.write(s + "\n")


def main():
    t = api(f"/repos/{REPO}/git/trees/main?recursive=1")
    tree = t.get("tree", [])
    total = {}
    for e in tree:
        total[e["path"]] = e.get("size", 0)
    log(f"仓库 {REPO}: {len(tree)} 个文件，共 {sum(total.values())/2**20:.0f} MiB")

    shas = {e["path"]: e["sha"] for e in tree}
    parts = sorted([p for p in total if p.startswith("model_part_")])
    log(f"分片 {len(parts)} 个")

    dst = OUT / "model.safetensors"
    done = set()
    mark = OUT / "_done.json"
    if mark.exists():
        done = set(json.loads(mark.read_text()))
    log(f"已完成 {len(done)}/{len(parts)}")

    mode = "ab" if dst.exists() and len(done) < len(parts) else "wb"
    if not dst.exists():
        mode = "wb"

    t0 = time.time()
    with open(dst, mode) as f:
        for i, p in enumerate(parts):
            if p in done:
                continue
            # ★ 用 API blob 端点（Accept: raw），实测 8 MiB/s
            #   raw.githubusercontent.com 只有 0.01 MiB/s（慢 800 倍）
            sha = shas[p]
            req = urllib.request.Request(
                f"{API}/repos/{REPO}/git/blobs/{sha}",
                headers={"Authorization": f"Bearer {TOKEN}",
                         "Accept": "application/vnd.github.raw",
                         "User-Agent": "t/1"})
            data = urllib.request.urlopen(req, timeout=180).read()
            exp = total[p]
            if len(data) != exp:
                log(f"  ✗ {p} 大小不符 {len(data)} != {exp}")
                sys.exit(1)
            f.write(data)
            done.add(p)
            mark.write_text(json.dumps(sorted(done)))
            got = sum(total[x] for x in done)
            el = time.time() - t0
            sp = got / max(el, 1) / 2**20
            log(f"  [{len(done):>2}/{len(parts)}] {p}  "
                f"{got/2**20:>7.0f}/{sum(total[x] for x in parts)/2**20:.0f} MiB  "
                f"{sp:.1f} MiB/s  [{el:.0f}s]")

    # 小文件
    for name in ("config.json", "model.safetensors.index.json",
                 "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "generation_config.json",
                 "chat_template.jinja"):
        o = OUT / name
        if o.exists() and o.stat().st_size > 0:
            continue
        try:
            req = urllib.request.Request(
                f"{API}/repos/{REPO}/git/blobs/{shas[name]}",
                headers={"Authorization": f"Bearer {TOKEN}",
                         "Accept": "application/vnd.github.raw",
                         "User-Agent": "t/1"})
            o.write_bytes(urllib.request.urlopen(req, timeout=60).read())
            log(f"  ✓ {name}")
        except Exception as e:
            log(f"  - {name} {str(e)[:40]}")

    size = dst.stat().st_size
    want = sum(total[x] for x in parts)
    log(f"\n✅ {dst}  {size:,} B  期望 {want:,}  {'一致' if size==want else '★不符★'}")
    log(f"   耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
