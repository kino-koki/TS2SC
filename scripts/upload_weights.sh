#!/usr/bin/env bash
# ===================================================================
# 把本地已下载的权重推到 GitHub（普通 git 对象，不用 LFS）
# ===================================================================
# 为什么必须这样：
#   * GitHub 单文件硬限 100MB → 必须切分
#   * Git LFS 免费账户只有 1GB → 2GB 权重会超额
#   * Release 附件在我的沙箱里**只能传不能收**（403）
#   * 只有 git 仓库内容是双向可达的（codeload / raw.githubusercontent 通）
# ===================================================================
set -euo pipefail

# ---------- 你只需要改这三行 ----------
SRC_DIR="$HOME/MiniCPM5-1B"          # 本地权重目录
REPO="kino-koki/minicpm5-1b-weights" # 目标仓库（先去 GitHub 建好，可私有）
CHUNK="20M"                          # 分片大小，必须 <100MB，建议 20M
# --------------------------------------

TOKEN="${GH_TOKEN:-}"   # export GH_TOKEN=ghp_xxx
[ -n "$TOKEN" ] || { echo "请先: export GH_TOKEN=ghp_xxx"; exit 1; }

echo "==> 源目录: $SRC_DIR"
ls -la "$SRC_DIR" | head -20

TMP="$(mktemp -d)/pushrepo"
mkdir -p "$TMP"
cd "$TMP"
git init -q
git config user.name kino-koki
git config user.email dev@tuan.local
git config http.postBuffer 524288000
git remote add origin "https://kino-koki:${TOKEN}@github.com/${REPO}.git"

# ---- 1. 小文件（json / txt / py）直接进 ----
find "$SRC_DIR" -maxdepth 1 -type f \
     \( -name "*.json" -o -name "*.txt" -o -name "*.py" -o -name "*.model" \) \
     -size -95M -exec cp {} . \; 2>/dev/null || true

# ---- 2. 大文件（safetensors / bin / gguf）切分 ----
mkdir -p parts
for f in "$SRC_DIR"/*.safetensors "$SRC_DIR"/*.bin "$SRC_DIR"/*.gguf; do
  [ -e "$f" ] || continue
  base="$(basename "$f")"
  echo "  → 切分 $base"
  split -b "$CHUNK" -d -a 3 "$f" "parts/${base}.part"
done
ls parts | wc -l | xargs echo "  分片数:"

# ---- 3. 记录切分方式，方便还原 ----
cat > RESTORE.sh <<'EOS'
#!/usr/bin/env bash
# 还原：cat parts/<name>.part* > <name>
set -e
mkdir -p out
for base in $(ls parts | sed 's/\.part[0-9]*$//' | sort -u); do
  cat parts/${base}.part* > out/${base}
  echo "  restored: $base"
done
EOS
chmod +x RESTORE.sh

# ---- 4. 推 ----
git add -A
git commit -q -m "MiniCPM5-1B 权重（git 对象分片，非 LFS）

分片 ${CHUNK}，还原见 RESTORE.sh"
git branch -M main
echo "==> 推送中（2GB 大约几分钟）..."
git push -u origin main

echo ""
echo "==> 完成: https://github.com/${REPO}/tree/main"
echo "==> 告诉我仓库名，我立刻拉取"
