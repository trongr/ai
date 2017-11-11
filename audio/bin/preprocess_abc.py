# USAGE. python preprocess_abc.py master.abc > master_preprocessed.abc

from __future__ import print_function
import sys

INPUT_FILENAME = sys.argv[1]

with open(INPUT_FILENAME, "r", encoding="ascii", errors="surrogateescape") as f:
    for line in f:
        if not line.startswith(("%", "w:", "A:", "B:", "C:", "D:", "F:", "G:", "H:", "I:",
            "N:", "O:", "R:", "S:", "T:", "Z:", "W:")):
            print(line, end="")