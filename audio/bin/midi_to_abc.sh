# USAGE
# bash midi_2_abc.sh SRC_DIR_1 SRC_DIR_2 ...
# 
# Converts all midi files (ending in .mid) in source dirs into abc files in the 
# same directories.

SRC_DIR="$@"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILES=$(find $SRC_DIR -type f | grep "\.mid")
MIDI2ABC=$SCRIPT_DIR/abctools-win-20170826/midi2abc.exe

for file in $FILES; do
    output=$(basename "$file" .mid).abc
    $MIDI2ABC $file > $output
done