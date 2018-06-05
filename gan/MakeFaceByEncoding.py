import Faces
import sys
encoding = sys.argv[1]
outputDir = sys.argv[2]
imgFilename = sys.argv[3]
txtFilename = sys.argv[4]
encoding = [float(x) for x in encoding.split(",")]
Faces.MakeFaceByEncoding(encoding, outputDir, imgFilename, txtFilename)
