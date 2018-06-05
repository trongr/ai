#!/bin/bash
# Need this bash script cause we have to activate the virtualenv
source activate tensorflow
python MakeFaceByEncoding.py "$1" "$2" "$3" "$3"