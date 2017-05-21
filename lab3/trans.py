from PIL import Image
import sys

im = Image.open(sys.argv[1])
im.save(sys.argv[2])
