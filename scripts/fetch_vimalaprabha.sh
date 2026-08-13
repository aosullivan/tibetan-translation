#!/usr/bin/env bash
# fetch_vimalaprabha.sh — locate and extract the Vimalaprabha (Toh. 1347)
# Wylie text from the ACIP Release 6 archive.
set -uo pipefail

UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
ZIP_URL='https://resources.christian-steinert.de/download/acip-release6-wylie.zip'
BASE='https://acip-archive.christian-steinert.de/wylie'
WORK="$(pwd)/acip_work"
mkdir -p "$WORK"; cd "$WORK"

say() { printf '\n=== %s ===\n' "$*"; }

# Patterns that identify the text. ACIP marks the Derge number; the Tibetan
# title string is the reliable content match.
TITLE_RE="dri ma med pa'i 'od"
TITLE_RE2="dus kyi 'khor lo'i 'grel chen"

# ---------- Route 1: the single zip (one request, no rate limit) ----------
say "Route 1: trying the full Wylie zip"
if [ ! -f acip-release6-wylie.zip ]; then
  curl -L --compressed -A "$UA" -o acip-release6-wylie.zip "$ZIP_URL"
fi

if unzip -tq acip-release6-wylie.zip >/dev/null 2>&1; then
  echo "Zip is valid. Extracting..."
  unzip -oq acip-release6-wylie.zip -d zip_extract
  HITS=$(grep -rlF "$TITLE_RE" zip_extract 2>/dev/null)
  [ -z "$HITS" ] && HITS=$(grep -rlF "$TITLE_RE2" zip_extract 2>/dev/null)
  if [ -n "$HITS" ]; then
    echo "Found in:"; echo "$HITS"
    echo "$HITS" > ../vimalaprabha_files.list
    echo "File list written to vimalaprabha_files.list"
    echo "Run the extraction section below on these files."
    exit 0
  else
    echo "Zip extracted but no title match — check zip_extract/ manually."
  fi
else
  echo "Zip missing or corrupt (likely an error page was saved). Size:"
  ls -la acip-release6-wylie.zip 2>/dev/null
  echo "First bytes:"; head -c 200 acip-release6-wylie.zip 2>/dev/null; echo
fi

# ---------- Route 2: per-file, polite, with diagnostics ----------
say "Route 2: per-file fetch of rgyud volumes (TD10001-10078)"
ok=0; bad=0
for n in $(seq 10001 10078); do
  f="rgyud_${n}.txt"
  curl -s --compressed -A "$UA" -e "$BASE/" -o "$f" "$BASE/TD${n}_T.TXT"
  sz=$(wc -c < "$f")
  if [ "$sz" -lt 500 ]; then
    bad=$((bad+1))
    if [ "$bad" -le 1 ]; then
      echo "Small response on TD${n} (${sz} bytes). Body says:"
      cat "$f"; echo
    fi
  else
    ok=$((ok+1))
  fi
  sleep 1   # be polite; avoids rate-limit throttling
done
echo "Got $ok real files, $bad small/error responses."

if [ "$ok" -eq 0 ]; then
  echo "All responses were stubs. Read the body printed above:"
  echo " - 'Host not in allowlist' => network egress block on YOUR side."
  echo " - '403 / forbidden'       => server guard; try a browser session."
  echo " - 'rate' / 'many'         => throttled; increase sleep, rerun later."
  exit 1
fi

say "Searching downloaded volumes for the Vimalaprabha"
HITS=$(grep -lF "$TITLE_RE" rgyud_*.txt 2>/dev/null)
[ -z "$HITS" ] && HITS=$(grep -lF "$TITLE_RE2" rgyud_*.txt 2>/dev/null)
if [ -z "$HITS" ]; then
  echo "No title match in TD10001-10078. The text may sit outside this range."
  echo "Largest files (likely multi-text rgyud volumes) to inspect:"
  wc -c rgyud_*.txt | sort -n | tail
  exit 1
fi
echo "Found in:"; echo "$HITS"
echo "$HITS" > ../vimalaprabha_files.list

say "Context around first hit (so we can see the internal delimiters)"
first=$(echo "$HITS" | head -1)
grep -n "$TITLE_RE" "$first" | head
echo "--- 40 lines around it ---"
ln=$(grep -n "$TITLE_RE" "$first" | head -1 | cut -d: -f1)
sed -n "$((ln>20?ln-20:1)),$((ln+20))p" "$first"
