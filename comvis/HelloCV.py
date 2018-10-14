import sys
import sys
from PIL import Image
from pytesseract import image_to_string
import cv2

filepath = sys.argv[1]
video = cv2.VideoCapture(filepath)

FRAME_RATE = int(video.get(cv2.CAP_PROP_FPS))
print("FRAME RATE", FRAME_RATE)

frame = -1
success = True

while success:
    frame += 1

    # Only extract text every second.
    if frame % FRAME_RATE != 0:
        continue

    video.set(1, frame)
    success, image = video.read()

    print("%s - %d" % (filepath, frame / FRAME_RATE))

    if not success:
        print("VIDEO READ FAILED")
        break

    # cv2.imwrite("%s-%d.jpg" % (filepath, frame), image)

    print("============================================================")
    print(unicode(image_to_string(image, lang='eng')).encode('utf8'))
    print("============================================================")

video.release()
