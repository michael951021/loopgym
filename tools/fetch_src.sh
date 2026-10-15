#!/usr/bin/env bash
# Pinned third-party sources for API checking (CLAUDE.md rule 6).
# For vllm, deepspeed, trl: fetch the latest stable (non-prerelease) PyPI release
# (wheel, or sdist when there is no wheel) with curl, unpack it read-only into
# .tools/src/<name>-<version>/, and record the exact version in tools/versions.txt.
# Never installed into .venv. Idempotent: a re-run skips a package whose recorded
# version is already unpacked.
set -euo pipefail
cd "$(dirname "$0")/.."

S=.tools/src
mkdir -p "$S"

fetch() { curl -fsSL --retry 3 "$@"; }

record() {
  local k=$1 v=$2
  if grep -q "^$k=" tools/versions.txt; then
    sed -i "s|^$k=.*|$k=$v|" tools/versions.txt
  else
    printf '%s=%s\n' "$k" "$v" >> tools/versions.txt
  fi
}

if ! grep -q '^vllm=' tools/versions.txt; then
  printf '\n# Pinned third-party sources (read-only copies under .tools/src/, tools/fetch_src.sh)\n' >> tools/versions.txt
fi

for name in vllm deepspeed trl; do
  echo "== $name"
  # Honor the pin in tools/versions.txt so a fresh clone fetches the exact pinned
  # version (CLAUDE.md rule 6: pinned versions only); latest is only a first-run fallback.
  PIN=$(sed -n "s/^$name=//p" tools/versions.txt | head -1)
  work=$S/.stage-$name
  rm -rf "$work"
  mkdir -p "$work"
  fetch "https://pypi.org/pypi/$name/json" -o "$work/meta.json"
  line=$(python3 - "$work/meta.json" "${PIN:-}" <<'EOF'
import json, re, sys
d = json.load(open(sys.argv[1]))
def pre(v):
    return '+' in v or re.search(r'\.(a|b|rc|dev)\d*$', v, re.I) is not None
cands = [v for v in d['releases'] if v.strip() and not pre(v)]
if not cands:
    sys.exit('no stable release for ' + d['info']['name'])
def key(v):
    return (tuple(int(x) for x in re.match(r'\d+(?:\.\d+)*', v).group(0).split('.')), v)
pin = sys.argv[2] if len(sys.argv) > 2 else ''
if pin:
    if pin not in d['releases']:
        sys.exit(f"pinned {d['info']['name']}=={pin} not found on PyPI")
    ver = pin
else:
    ver = max(cands, key=key)
def score(f):
    n = f['filename']
    if f['packagetype'] == 'bdist_wheel':
        return (0, 'py3-none-any' not in n, n)
    if f['packagetype'] == 'sdist':
        return (2, 0, n)
    return (3, 0, n)
f = min(d['releases'][ver], key=score)
print(ver, f['url'], f['filename'])
EOF
)
  ver=${line%% *}; rest=${line#* }; url=${rest%% *}; fname=${rest#* }
  dst=$S/$name-$ver
  if [ -d "$dst" ] && grep -q "^$name=$ver$" tools/versions.txt; then
    echo "   already fetched $ver at $dst"
    rm -rf "$work"
    continue
  fi
  find "$dst" -mindepth 1 -delete 2>/dev/null || true
  mkdir -p "$dst"
  fetch "$url" -o "$work/$fname"
  case $fname in
    *.whl)
      python3 -m zipfile -e "$work/$fname" "$dst/"
      ;;
    *.tar.gz)
      tar -xzf "$work/$fname" -C "$work"
      # Flatten a single top-level dir (sdist convention) into $dst.
      top=$(find "$work" -mindepth 1 -maxdepth 1 ! -name meta.json ! -name "$fname")
      if [ "$(printf '%s\n' "$top" | wc -l)" -eq 1 ] && [ -d "$top" ]; then
        find "$top" -mindepth 1 -maxdepth 1 -exec mv {} "$dst"/ \;
      else
        printf '%s\n' "$top" | while IFS= read -r t; do [ -n "$t" ] && mv "$t" "$dst"/; done
      fi
      ;;
    *)
      echo "FATAL: unexpected artifact type for $name: $fname" >&2
      exit 1
      ;;
  esac
  rm -rf "$work"
  record "$name" "$ver"
  echo "   $fname -> $dst ($(du -sh "$dst" | cut -f1))"
done
echo "fetched sources: $(find "$S" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -printf '%f ' | sed 's/ $//')"
