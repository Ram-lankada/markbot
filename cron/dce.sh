#!/bin/bash
# Info: https://github.com/Tyrrrz/DiscordChatExporter/blob/master/.docs

set -euo pipefail

TOKEN="${TOKEN:-NzAwMzI4Mjc1Njc4OTg2Mjcw.GBb9MK.3Clap5mFzd_3ayxyp2dFbtLbD-kb_43q9UeQQ8}"          # Better: export TOKEN in env or .env, don’t hardcode
CHANNELID="${CHANNELID:-1433469014662840421}"

DLLFOLDER="/home/ram/Documents/markbot/discord_tech_map/bin"
FILENAME="dce_output"
EXPORT_DIRECTORY="../exports"
EXPORTFORMAT="json" 
# Available export formats: plaintext, htmldark, htmllight, json, csv
# /\ CaSe-SeNsItIvE /\
# You can edit the export command on line 40 if you'd like to include more options like date ranges and date format. You can't use partitioning (-p) with this script.

# This will verify if EXPORTFORMAT is valid and will set the final file extension according to it. If the format is invalid, the script will display a message and exit.
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
echo "Available export formats: plaintext, htmldark, htmllight, csv, json"
echo "/\ CaSe-SeNsItIvE /\\"
exit 1
fi

# This will change the script's directory to DLLPATH, if unable to do so, the script will exit.
cd $DLLFOLDER || exit 1

# This will export your chat
./DiscordChatExporter.Cli export -t $TOKEN -c $CHANNELID -f $EXPORTFORMAT -o $FILENAME.tmp

# This sets the current time to a variable
CURRENTTIME=$(date +"%Y-%m-%d-%H-%M-%S")

# This will move the .tmp file to the desired export location, if unable to do so, it will attempt to delete the .tmp file.
if ! mv "$FILENAME.tmp" "${EXPORT_DIRECTORY//\"}/$FILENAME-$CURRENTTIME$FORMATEXT" ; then
echo "Unable to move $FILENAME.tmp to $EXPORT_DIRECTORY/$FILENAME-$CURRENTTIME$FORMATEXT."
echo "Cleaning up..."
  if ! rm -Rf "$FILENAME.tmp" ; then
  echo "Unable to remove $FILENAME.tmp."
  fi
exit 1
fi
exit 0