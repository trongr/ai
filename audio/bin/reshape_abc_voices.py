"""
USAGE
=====
python reshape_abc_voices.py master_tmp.abc > master.abc

This script takes a file containing multiple ABC songs each with potentially
more than one voices, and outputs the same songs with shorter and more segments
for each voice by interleaving them. An ABC song can list all notes in one
voice, then all notes in another voice, which makes it harder for a network to
learn. It's easier to learn if the song were to list a short line of notes in
one voice, then another short line of notes in another voice, then back to the
first voice, and so on.
"""

import sys

INPUT_FILENAME = sys.argv[1]

"""
song is a string containing a single ABC song. Returns the song with voices
reshaped.
"""
def reshape_one_song(song):
    """
    STRUCTURE OF AN ABC FILE
    ========================
    An ABC file has the following format: header / metadata lines starting with
    capital letters and a colon, e.g. X:, K:, etc. Then zero, one or more voices
    denoted by V:. (Some songs omit V:# if they have only one voice.)
    """
    header = "" # Append header lines here
    voices = {} # {"V:1": ["One line", "Another line",...], "V:2": [...]}
    # Content of voices altogether in a string. We'll print header + voices_str
    # to stdout:
    voices_str = "" 

    """
    current_voice changes to the name of the voice once we see the first voice.
    If the song omits V:#, then current_voice will never change, and header will
    contain both the header and the one voice. That's OK, we'll just print out
    the header all the same.
    """
    current_voice = "header" 
    song = song.strip().split("\n")
    for line in song:
        line = line + "\n"
        if line.startswith("V:"):
            current_voice = line 
        elif current_voice is "header":
            header += line 
        else:
            voices[current_voice] = voices.get(current_voice, [])
            voices[current_voice].append(line)

    voices_max_length = 0
    for key in voices:
        voice_length = len(voices[key])
        if voice_length > voices_max_length:
            voices_max_length = voice_length

    for i in range(voices_max_length):
        for key in voices:
            line = ""
            try:
                line = voices[key][i]
            except:
                line = ""
            voices_str += key + line

    return header + voices_str

# Try this on Windows:
# with open(INPUT_FILENAME, "r", encoding="ascii", errors="surrogateescape") as f:
with open(INPUT_FILENAME, "r") as f:
    current_song = ""
    for line in f:
        if line.strip().startswith("X:"):
            print(reshape_one_song(current_song))
            current_song = line
        else:
            current_song += line

    print(reshape_one_song(current_song))