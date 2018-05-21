#!/bin/bash
# Need this bash script cause we have to activate the virtualenv
source activate tensorflow
python RandomFaces.py "$1" "$2" "$3"