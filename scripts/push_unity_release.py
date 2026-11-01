#!/usr/bin/env python3
"""把 SimplUnity 融合权重推到 GitHub Release（Release 附件，不用 LFS）"""
import json, os, sys, time, urllib.request, urllib.error
from pathlib import Path

TOKEN = os.environ["GH_TOKEN"]
REPO = "kino-koki/TS2SC"
TAG = os.getenv("REL_TAG", "v1.0-unity")
SRC = Path("/data/workspace/UNITY2")
API = "https://api.github.com"
H = {"Authorization": f"Bearer {TOKEN}",
     "Accept": "application/vnd.github+json", "User-Agent": "ts2sc/1"}


def api(path, method="GET", data=None, timeout=60):
    b = json.dumps(data).encode() if data is not None else None
    h = dict(H)
    if b:
        h["Content-Type"] = "application/json"
    r = urllib.request.Request(API + path, data=b, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as x:
            return x.status, json.loads(x.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:300]}
    except Exception as e:
        return None, {"error": f"{type(e).__name__}:{str(e)[:120]}"}


def raw_upload(url, path, name):
    data = path.read_bytes()
    req = urllib.request.Request(
        f"{url}?name={urllib.parse.quote(name)}", data=data,
        headers={"Authorization": f"Bearer {TOKEN}",
                 "Content-Type": "application/octet-stream",
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "ts2sc/1"}, method="POST")
    with urllib.request.urlopen(req, timeout=900) as x:
        return json.loads(x.read())


def main():
    import urllib.parse
    st, rel = api(f"/repos/{REPO}/releases/tags/{TAG}")
    if st != 200:
        st, rel = api(f"/repos/{REPO}/releases", "POST", {
            "tag_name": TAG, "name": "SimplUnity 融合模型 1.439B",
            "body": ("MiniCPM5-1B × Qwen3-0.6B → SimplCore 融合\n\n"
                     "- 1,438,585,652 参数（1.439 B）\n"
                     "- 81.8% 逐位真实预训练权重（263 张量全等验证）\n"
                     "- O(1) 流式状态（跨 L 恒定），4M 上下文 20.41 MiB\n"
                     "- 流式/并行对偶 max|Δ| = 1.01e-06\n"),
            "draft": False, "prerelease": False})
    if st not in (200, 201):
        print(f"✗ release {st} {rel.get('error','')[:200]}"); return
    up = rel["upload_url"].split("{")[0]
    print(f"Release {TAG}  id={rel['id']}")

    files = sorted(SRC.glob("model-*.safetensors"))
    files += [SRC / "config.json", SRC / "model.safetensors.index.json",
              SRC / "result.json", SRC / "verify.log",
              SRC / "complete.log", SRC / "unity.log"]
    for f in SRC.glob("ALIGN/*"):
        files.append(f)
    print(f"待传 {len(files)} 个文件")

    ok = 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        name = f.relative_to(SRC).as_posix().replace("/", "__")
        try:
            raw_upload(up, f, name)
            ok += 1
            el = time.time() - t0
            print(f"  [{i}/{len(files)}] {name}  {f.stat().st_size/2**20:.0f} MiB  "
                  f"[{el:.0f}s]", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(files)}] ✗ {name} {type(e).__name__}:{str(e)[:100]}",
                  flush=True)
    print(f"\n✅ {ok}/{len(files)} 上传完成，耗时 {time.time()-t0:.0f}s")
    print(f"https://github.com/{REPO}/releases/tag/{TAG}")


if __name__ == "__main__":
    main()
