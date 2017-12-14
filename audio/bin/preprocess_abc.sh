#!/bin/bash

# USAGE. bash preprocess_abc.sh input/file.abc output/file.abc
#
# This script preprocesses an ABC file, similar to how we generate the
# master.abc file for training, except this only does the preprocessing and
# reshaping.

SRC_DIR="$@"
# Windows
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")")"
# Mac
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

input_abc="$1"
output_abc="$2"

echo "Preprocessing...."
echo "Removing metadata"
python $SCRIPT_DIR/preprocess_abc.py $input_abc > master_tmp_preprocessed.abc

echo "Reshaping voices"
python $SCRIPT_DIR/reshape_abc_voices.py master_tmp_preprocessed.abc > $output_abc

echo "Removing tmp file"
rm master_tmp_preprocessed.abc