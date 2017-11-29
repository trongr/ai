# USAGE
# bash abc_to_midi.sh SRC_DIR_1 SRC_DIR_2 ...
# 
# Converts all midi files (ending in .mid) in source dirs into abc files in the 
# same directories.

SRC_DIR="$@"
FILES=$(find $SRC_DIR -type f | grep "\.abc")

# windows. For some reason this doesn't need pwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")")"
MIDI2ABC=$SCRIPT_DIR/abctools-win-20170826/abc2midi.exe 

# mac
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# MIDI2ABC=$SCRIPT_DIR/abcmidi/abc2midi 

for file in $FILES; do
    output=$(basename "$file" .abc).mid
    $MIDI2ABC $file
done