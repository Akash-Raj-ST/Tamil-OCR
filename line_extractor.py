import pytesseract as pt
from PIL import Image

img = Image.open('./test/tamil_text.png')

extracted = pt.image_to_string(img)

print(extracted)
