"""
USAGE. python randomize_training_data.py master_preprocessed.abc > master.abc

This script replaces song titles in an input abc file with increasing numbers,
because most songs in a training set have the same titles, e.g. X: 0, 1, 2,...,
which confuses the network into creating songs with duplicate titles, so when it
comes time to convert the output abc file into MIDI songs, those songs with
duplicate titles are overwritten by the last song with the same title, and we
have way fewer songs than we should. Bad.

The other thing this script does is randomize the songs, so that when we train
the network on a large dataset, it's not trained only on say waltz for a long
time (which would make it output waltz songs) and then trained only on baroque
songs for a long time (which would make it output baroque songs), and so on.
Instead, it'll be trained on randomized genres, so its output will be a mix. If
we want to generate waltz specifically, we should train it on only waltz songs
(optionally after it's been trained on a broad range of genres first).
"""

import sys
import re
from random import shuffle

INPUT_FILENAME = sys.argv[1]

regex = re.compile(r"^\s*X:.*\n")
songs = []
current_song = ""

# Add lines into songs
with open(INPUT_FILENAME, "r") as f:
    for line in f:
        if regex.match(line):
            songs.append(current_song)
            current_song = line
        else:
            current_song += line

shuffle(songs)

# Replace titles with increasing sequence
song_i = 0
for song in songs:
    song = regex.sub("X: " + str(song_i) + "\n", song)
    song_i += 1    
    print(song)