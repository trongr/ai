#!/bin/bash

# USAGE. bash gen_master_abc.sh MY_SRC_DIR ANOTHER_DIR ...
#
# This script cats all ABC files in the src dirs and cats them together,
# preprocesses them, randomizes the training data, and retitles the songs,
# and dumps them in the current working dir's master.abc

SRC_DIR="$@"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Removing master*.abc"
rm master*.abc

echo "Concatenating abc files into master_tmp_all.abc"
cat $(find $SRC_DIR -type f | grep "\.abc") > master_tmp_all.abc

echo "Preprocessing master_tmp_all.abc by removing metadata"
python $SCRIPT_DIR/preprocess_abc.py master_tmp_all.abc > master_tmp_preprocessed.abc

echo "Randomizing preprocessed file and outputting master.abc"
python $SCRIPT_DIR/randomize_training_data.py master_tmp_preprocessed.abc > master.abc

# echo "Truncating training data to first 10000 lines"
# head -10000 master_tmp_random.abc > master.abc

echo "Removing tmp files"
rm master_tmp_*.abc