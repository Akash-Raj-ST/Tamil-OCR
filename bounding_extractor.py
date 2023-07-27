import cv2
import os
from PIL import Image

wr = 1

for cp in os.listdir("./words"):
    image = Image.open(f'./words/{cp}')
    x=image.size[0]
    y=image.size[1]

    if x<500 or y<150:
        new_image = image.resize((1000,1000))
        new_image.save(f'./words/{cp}')

    image = cv2.imread(f'./words/{cp}')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img = cv2.dilate(binr, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    blur = cv2.medianBlur(img, 3)

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_with_contours = image.copy()

    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    cv2.imshow('Image with Contours', image_with_contours)

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
    print(new_boundings)

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
    print(boundings)

    k=1

    try:
        for i in os.listdir(f"./letters/{wr}"):
            z = "./letters/"+str(wr)+"/"+str(i)
            os.remove(z)	

        os.rmdir(f"./letters/{wr}")

    except:
        pass

    os.mkdir(f"./letters/{wr}")

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
        cv2.imwrite(f'./letters/{wr}/{k}.jpg', roi)
        k+=1

    wr+=1

    cv2.waitKey(0)
cv2.destroyAllWindows()