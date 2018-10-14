import sys
from PIL import Image
from pytesseract import image_to_string

filepath = sys.argv[1]
print image_to_string(Image.open(filepath), lang='eng')