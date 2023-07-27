#Import the require libraries
import cv2
import os
from PIL import Image

#Change the resolution of the image
image = Image.open('./test/word1.jpg')
x=image.size[0]
y=image.size[1]

if x<500 or y<150:
    new_image = image.resize((1000,1000))
    new_image.save('./test/word1.jpg')

#IMAGE PREPROCESSING
image = cv2.imread('./test/word1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to convert to binary image
_, binr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Apply dilation and erosion to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
img = cv2.dilate(binr, kernel, iterations=1)
img = cv2.erode(img, kernel, iterations=1)

# Apply median blur to remove remaining noise
blur = cv2.medianBlur(img, 3)

#FIND THE COUNTOURS

# Apply thresholding to preprocess the image
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours in the image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boundings = []

for i, contour in enumerate(contours):
    z = []

    x, y, w, h = cv2.boundingRect(contour)
    
    z.append(x)
    z.append(y)
    z.append(w)
    z.append(h)
    
    boundings.append(tuple(z))

boundings.sort()

erase = []
new_boundings = []

for i in range(len(boundings)):
    for j in range(i+1,len(boundings)):
        x1 = boundings[i][0]+boundings[i][2]
        x2 = boundings[j][0]+boundings[j][2]

        if x2<x1:
            erase.append(boundings[i])
            erase.append(boundings[j])
            
            z = []

            y = boundings[i][1] + boundings[i][3] - boundings[j][1]

            z.append(boundings[i][0])
            z.append(boundings[j][1])
            z.append(boundings[i][2])
            z.append(y)
            
            new_boundings.append(tuple(z))

for i in boundings:
    if i not in erase:
        new_boundings.append(i)

new_boundings.sort()
boundings = new_boundings

words_diff = []
new_boundings_2 = []

for i in range(1,len(boundings)):
    words_diff.append(boundings[i][0]-(boundings[i-1][0]+boundings[i-1][2]))
    if words_diff[len(words_diff)-1] >=0:
        new_boundings_2.append(boundings[i-1])
    else:
        words_diff = words_diff[0:len(words_diff)-1]

new_boundings_2.append(boundings[i])

new_boundings_2.sort()
boundings = new_boundings_2

mx = max(words_diff)
mx = mx//10 *10

new_boundings_3 = []
c=0

for i in range(len(words_diff)):
    if words_diff[i]>=mx:
        z = []

        x = boundings[c][0] 
        w = boundings[i][0]+boundings[i][2]-boundings[c][0]

        y=10000
        h = 0
        
        for j in range(c,i+1):
            if boundings[j][1]<y:
                y = boundings[j][1]

            if(boundings[j][1]+boundings[j][3]) > h:
                h= boundings[j][1]+boundings[j][3]
        
        h = h - y

        z.append(x)
        z.append(y)
        z.append(w)
        z.append(h)

        new_boundings_3.append(tuple(z))        
        c = i+1

    if i==(len(words_diff)-1):
        z = []

        x = boundings[c][0] 
        w = boundings[i+1][0]+boundings[i+1][2]-boundings[c][0]

        y=10000
        h = 0
        
        for j in range(c,i+2):
            if boundings[j][1]<y:
                y = boundings[j][1]

            if(boundings[j][1]+boundings[j][3]) > h:
                h= boundings[j][1]+boundings[j][3]
        
        h = h - y

        z.append(x)
        z.append(y)
        z.append(w)
        z.append(h)

        new_boundings_3.append(tuple(z))        
        c = i+1

new_boundings_3.sort()
boundings = new_boundings_3

print(boundings)

k=1

for i in os.listdir("./words"):
    z = "./words/"+str(i)
    os.remove(z)	

os.rmdir("./words")
os.mkdir("./words")

for i in boundings:
    # Extract the bounding box
    x = i[0]
    y = i[1]
    w = i[2]
    h = i[3]

    # Crop the image

    if (y-10>=0) and (x-10>=0) and (y+h+10<=img.shape[0]) and (x+w+10<=img.shape[1]): 
        roi = img[y-10:y+h+10, x-10:x+w+10]

    elif (y-5>=0) and (x-5>=0) and (y+h+5<=img.shape[0]) and (x+w+5<=img.shape[1]): 
        roi = img[y-5:y+h+5, x-5:x+w+5]
    
    else:
        roi = img[y:y+h,x:x+w]
    
    # Save the cropped image
    cv2.imwrite(f'./words/{k}.jpg', roi)
    k+=1