#!/usr/bin/env bash
set -euo pipefail

# Load secrets/config
ENVFILE=".env"
set -a
source "$ENVFILE"
set +a


if [[ "$EXPORTFORMAT" == "plaintext" ]]; then
FORMATEXT=.txt
elif [[ "$EXPORTFORMAT" == "htmldark" ]] || [[ "$EXPORTFORMAT" == "htmllight" ]]; then
FORMATEXT=.html
elif [[ "$EXPORTFORMAT" == "json" ]]; then
FORMATEXT=.json
elif [[ "$EXPORTFORMAT" == "csv" ]]; then
FORMATEXT=.csv
else
echo "$EXPORTFORMAT - Unknown export format"
exit 1
fi

cd "$BIN_DIR"

TS="$(date +"%Y-%m-%d-%H-%M-%S")"

for CID in $CHANNEL_IDS; do
  echo "Exporting $CID at $TS..."
  ./DiscordChatExporter.Cli export \
    -t "$DISCORD_TOKEN" \
    -c "$CID" \
    -f "$EXPORTFORMAT" \
    -o "$EXPORT_DIR/channel_${CID}_${TS}.tmp"

    if ! mv -f "$EXPORT_DIR/channel_${CID}_${TS}.tmp" "$EXPORT_DIR/channel_${CID}_${TS}.${EXPORTFORMAT}" ; then 
    echo "Unable to move tempfile to json format" 
    echo "Cleaning Temp"
        if ! rm -Rf "$EXPORT_DIR/channel_${CID}_${TS}.tmp" ; then 
        echo "Unable to clean tempfile"
        fi
    exit 1 
    fi

done

echo "Done."

# command python3 "$(dirname "$0")/llm.py"