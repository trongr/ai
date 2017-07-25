#!/bin/bash

mkdir -p csv
rm csv/*

for filename in midi/*.mid; do
    filename_no_ext="$(basename "$filename" .mid)"
    midicsv midi/$filename_no_ext.mid csv/$filename_no_ext.csv
done

cat csv/* > csv/master.csv