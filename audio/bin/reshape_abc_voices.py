"""
USAGE
=====
python reshape_abc_voices.py input.abc output.abc

This script takes a file containing an ABC song with potentially more than one
voices, and outputs the same songs with shorter and more segments for each voice
by interleaving them. An ABC song can list all notes in one voice, then all
notes in another voice, which makes it harder for a network to learn. It's
easier to learn if the song were to list a short line of notes in one voice,
then another short line of notes in another voice, then back to the first voice,
and so on.
"""

