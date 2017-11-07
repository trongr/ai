#!/bin/bash

# USAGE. bash gen_master_abc.sh SRC_DIR DST_DIR

SRC_DIR="$1"
DST_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Removing $DST_DIR/master*.abc"
rm $DST_DIR/master*.abc

echo "Concatenating abc files into master_tmp_all.abc"
cat $(find $SRC_DIR -type f | grep "\.abc") > $DST_DIR/master_tmp_all.abc

echo "Preprocessing master_tmp_all.abc by removing metadata"
python $SCRIPT_DIR/preprocess_abc.py $DST_DIR/master_tmp_all.abc > $DST_DIR/master_tmp_preprocessed.abc

echo "Randomizing preprocessed file and outputting master.abc"
python $SCRIPT_DIR/randomize_training_data.py $DST_DIR/master_tmp_preprocessed.abc > $DST_DIR/master.abc

echo "Removing tmp files"
rm $DST_DIR/master_tmp_*.abc