#!/bin/bash
# dub.sh -- interactive terminal client for the dubbing service.
# Asks for the video and the languages, sends the job, waits, saves and plays the result.
#
#   bash dub.sh
#
# Optional: talk to the server directly on this Mac (faster for big videos):
#   DUB_URL=http://localhost:8000 bash dub.sh

URL="${DUB_URL:-https://remindful-making-stand.ngrok-free.dev}"
KEY="${DUB_KEY:-$(cat "$HOME/.tts_key" 2>/dev/null)}"
POLL="${DUB_POLL:-3}"
SKIP="ngrok-skip-browser-warning: true"

# read one field out of a JSON reply (needs python3, which every Mac with Homebrew has)
json_get() {
  PYTHONIOENCODING=utf-8 python3 -c 'import sys,json; d=json.load(sys.stdin); v=d.get(sys.argv[1],""); print(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False))' "$1" 2>/dev/null
}

echo "=============================================="
echo "  Video Dubbing  -  $URL"
echo "=============================================="

if [ -z "$KEY" ]; then
  read -r -s -p "API key: " KEY
  echo
fi

# 1. is the service reachable?
if ! curl -s -f -H "$SKIP" "$URL/health" >/dev/null; then
  echo "Cannot reach the service at $URL."
  echo "Is the server tab running, and (for the public address) the ngrok tab?"
  exit 1
fi

# 2. which video?
echo
read -r -p "Drag the video file into this window and press Enter: " VIDEO
VIDEO="${VIDEO//\\/}"                        # remove backslashes added by drag-and-drop
VIDEO="${VIDEO#"${VIDEO%%[![:space:]]*}"}"   # trim leading spaces
VIDEO="${VIDEO%"${VIDEO##*[![:space:]]}"}"   # trim trailing spaces
VIDEO="${VIDEO#\'}"; VIDEO="${VIDEO%\'}"     # strip quotes, if any
if [ ! -f "$VIDEO" ]; then
  echo "File not found: $VIDEO"
  exit 1
fi

# 3. which languages?
echo
echo "Language SPOKEN in the video:"
PS3="Choose a number: "
select NAME in Hindi Kannada Telugu Tamil English; do
  case "$NAME" in
    Hindi) SRC=hi;; Kannada) SRC=kn;; Telugu) SRC=te;; Tamil) SRC=ta;; English) SRC=en;;
    *) echo "Invalid choice, try again."; continue;;
  esac
  SRC_NAME="$NAME"
  break
done
[ -z "$SRC" ] && { echo "No language chosen."; exit 1; }

echo
echo "Language to DUB INTO:"
PS3="Choose a number: "
select NAME in Hindi Kannada Telugu Tamil; do
  case "$NAME" in
    Hindi) TGT=hi;; Kannada) TGT=kn;; Telugu) TGT=te;; Tamil) TGT=ta;;
    *) echo "Invalid choice, try again."; continue;;
  esac
  TGT_NAME="$NAME"
  break
done
[ -z "$TGT" ] && { echo "No language chosen."; exit 1; }

echo
echo "Video : $VIDEO"
echo "Dub   : $SRC_NAME  ->  $TGT_NAME"
read -r -p "Start? [Y/n] " OK
case "$OK" in n|N) echo "Cancelled."; exit 0;; esac

# 4. send the job
echo
echo "Uploading the video..."
RESP=$(curl -S --progress-bar -X POST "$URL/tts" \
  -H "x-api-key: $KEY" -H "$SKIP" \
  -F "source_lang=$SRC" -F "target_lang=$TGT" -F "video=@$VIDEO")
JOB=$(echo "$RESP" | json_get job_id)
if [ -z "$JOB" ]; then
  echo "The service did not accept the job. Its answer was:"
  echo "$RESP"
  exit 1
fi
echo "Job id: $JOB"

# 5. wait for it
START=$(date +%s)
FAILS=0
while true; do
  S=$(curl -s -H "x-api-key: $KEY" -H "$SKIP" "$URL/status/$JOB")
  ST=$(echo "$S" | json_get status)
  if [ -z "$ST" ]; then
    FAILS=$((FAILS + 1))
    if [ "$FAILS" -ge 5 ]; then
      echo
      echo "Lost contact with the service."
      exit 1
    fi
  else
    FAILS=0
    if [ "$ST" = "done" ] || [ "$ST" = "failed" ]; then
      break
    fi
    printf "\r  %s ... %ss      " "$ST" $(( $(date +%s) - START ))
  fi
  sleep "$POLL"
done
echo

if [ "$ST" = "failed" ]; then
  echo "The job failed: $(echo "$S" | json_get error)"
  exit 1
fi

echo "Finished in $(( $(date +%s) - START )) seconds."
echo
REF_TXT=$(echo "$S" | json_get ref_text)
DUB_TXT=$(echo "$S" | json_get translated_text)
echo "Text spoken in the video : $REF_TXT"
echo "Dubbed text ($TGT_NAME)  : $DUB_TXT"

# 6. save the result
OUTDIR="$HOME/Desktop"
[ -d "$OUTDIR" ] || OUTDIR="$PWD"
OUT="$OUTDIR/dubbed_${SRC}_to_${TGT}_${JOB:0:6}.wav"
if ! curl -s -f -H "x-api-key: $KEY" -H "$SKIP" -o "$OUT" "$URL/result/$JOB"; then
  echo "Download failed."
  exit 1
fi
echo
echo "Saved: $OUT"


# 7. play it (original clip first, if it is on this Mac, then the dub)
if command -v afplay >/dev/null 2>&1; then
  if [ -f "jobs/$JOB/audio.mp3" ]; then
    echo
    echo "Playing the ORIGINAL clip ($SRC_NAME)..."
    afplay "jobs/$JOB/audio.mp3"
  fi
  echo
  echo "Playing the DUBBED audio ($TGT_NAME)..."
  afplay "$OUT"
fi
echo
echo "Done."