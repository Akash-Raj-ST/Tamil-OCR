import cv2
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import gtts as gt

from tensorflow.keras.models import load_model

class TamilImageDetection:
    str = ""
    _str_arr = []
    length = 0

    vallinam=0
    mellinam=0
    idayinam=0

    uyir=0
    mei = 0
    uyir_mei = 0
    ayutha = 0

    total_characters=0

    para = ""
    eg = ""

    #uyir_list eluthu
    uyir_list = list('அஆஇஈஉஊஎஏஐஒஓஔஃ')

    #Mei eluthu
    mei_list = ["க","ங","ச","ஞ","ட","ண","த","ந","ப","ம","ய","ர","ல","வ","ழ","ள","ற","ன"]

    #vallinam_list
    vallinam_list = ["க","ச","ட","த","ப","ற"]

    #Mellinam
    mellinam_list = ["ங","ஞ","ண","ந","ம","ன"]

    #Idayinam
    idayinam_list = ["ய","ர","ல","வ","ழ","ள"]

    #Ascii values for the characters like "்", "ெ", "ோ", "ீ"
    asc = [3021,3006,3007,3008,3009,3010,3014,3015,3016,3018,3019,3020]
    
    def __init__(self,loc,vl):

        if vl==1:
            self.para = loc

        elif vl==0:
            #check whether image exists
            self.image_loc = loc
            self.line_extract_loc = "./lines"
            self.word_extract_loc = "./words"
            self.char_extract_loc = "./characters"

            self.model = load_model(os.path.join('models','imageclassifier.h5'))

            self.letters = pd.read_csv("./letters.txt",delimiter=" ")

            image = cv2.imread(self.image_loc)
            cv2.imshow(f'Main Image',image)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def lineExtract(self):

        #IMAGE PREPROCESSING
        image = cv2.imread(self.image_loc)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to the image
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        # Apply dilation and erosion to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        # Find the contours in the image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_with_contours = image.copy()

        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

        boundings = []

        for i, contour in enumerate(contours):
            z = []

            x, y, w, h = cv2.boundingRect(contour)
            
            z.append(x)
            z.append(y)
            z.append(w)
            z.append(h)
            
            boundings.append(tuple(z))

        boundings_3 = []

        for i in boundings:
            if i[3]<=10 and i[2]<=10:
                continue
            else:
                boundings_3.append(i)

        temp = boundings
        boundings = boundings_3
        boundings_3 = temp

        boundings.sort(key=lambda x:x[1])

        new_boundings = []

        k=0

        for i in range(1,len(boundings)):
            j = i-1
            
            y1 = boundings[i][1]
            y2 = boundings[j][1]+boundings[j][3]

            if y1>y2:
                new_boundings.append(tuple(boundings[k:i]))
                k=i

        new_boundings.append(tuple(boundings[k:i]))
            
        temp = []
        boundings_2 = []

        for i in new_boundings:
            x=2000
            y=2000
            w=0
            h=0

            temp = []

            for j in i:
                if j[0]<x:
                    x=j[0]
                if j[1]<y:
                    y=j[1]
                if w<(j[0]+j[2]):
                    w = j[0]+j[2]
                if h<(j[1]+j[3]):
                    h = j[1]+j[3]

            temp.append(x)
            temp.append(y)
            temp.append(w)
            temp.append(h)

            boundings_2.append(tuple(temp))

        try:
            for i in os.listdir(f"{self.line_extract_loc}"):
                z = f"{self.line_extract_loc}/"+str(i)
                os.remove(z)	
        except:
            pass

        os.rmdir(f"{self.line_extract_loc}")
        os.mkdir(f"{self.line_extract_loc}")

        wr = 1

        for i in boundings_2:
            x = i[0]
            y = i[1]
            w = i[2]
            h = i[3]

            if (y-10>=0) and (x-10>=0) and (h+10<=image.shape[0]) and (w+10<=image.shape[1]): 
                roi = image[y-10:h+10, x-10:w+10]

            elif (y-5>=0) and (x-5>=0) and (h+5<=image.shape[0]) and (w+5<=image.shape[1]): 
                roi = image[y-5:h+5, x-5:w+5]
            else:
                roi = image[y:h,x:w]
            
            print(i)

            roi = cv2.resize(roi,(1000,200))
            cv2.imwrite(f'{self.line_extract_loc}/{wr}.jpg',roi)

            wr+=1

        cv2.destroyAllWindows()

    def wordExtract(self):
        print("Extracting word...")
        wr = 1

        try:
            for i in os.listdir(f"{self.word_extract_loc}"):
                z = f"{self.word_extract_loc}/"+str(i)
                os.remove(z)	
        except:
            pass

        try:
            os.rmdir(f"{self.word_extract_loc}")
        except:
            pass

        os.mkdir(f"{self.word_extract_loc}")

        for cp in os.listdir(f"{self.line_extract_loc}"): 

        #IMAGE PREPROCESSING
            image = cv2.imread(f'{self.line_extract_loc}/{cp}')

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply a threshold to the image
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

            # Apply dilation and erosion to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            thresh = cv2.erode(thresh, kernel, iterations=1)

            # Find the contours in the image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            image_with_contours = image.copy()

            cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
            cv2.imshow(f'line', image_with_contours)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

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
            mx = mx - 6

            print(words_diff)
            print(mx)

            new_boundings_3 = []
            c=0

            pqr = 0

            for i in range(len(words_diff)):
                if words_diff[i]>=mx:
                    pqr=1
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

            for i in boundings:
                # Extract the bounding box
                x = i[0]
                y = i[1]
                w = i[2]
                h = i[3]

                # Crop the image

                if (y-10>=0) and (x-10>=0) and (y+h+10<=image.shape[0]) and (x+w+10<=image.shape[1]): 
                    roi = image[y-10:y+h+10, x-10:x+w+10]

                elif (y-5>=0) and (x-5>=0) and (y+h+5<=image.shape[0]) and (x+w+5<=image.shape[1]): 
                    roi = image[y-5:y+h+5, x-5:x+w+5]

                else:
                    roi = image[y:y+h,x:x+w]

                # Save the cropped image
                roi = cv2.resize(roi,(700,200))
                cv2.imwrite(f'{self.word_extract_loc}/{wr}.jpg', roi)
                wr+=1

    def characterExtract(self):
        wr = 1

        for cp in range(len(os.listdir(f"{self.word_extract_loc}"))):
            image = cv2.imread(f'{self.word_extract_loc}/{cp+1}.jpg')

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            _, binr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            img = cv2.dilate(binr, kernel, iterations=1)
            img = cv2.erode(img, kernel, iterations=1)

            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            image_with_contours = thresh.copy()

            cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

            cv2.imshow('word', image_with_contours)

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
            print(boundings)

            k=1

            try:
                for i in os.listdir(f"{self.char_extract_loc}/{wr}"):
                    z = f"{self.char_extract_loc}/"+str(wr)+"/"+str(i)
                    os.remove(z)	

                os.rmdir(f"{self.char_extract_loc}/{wr}")

            except:
                pass

            try:
                os.mkdir(f"{self.char_extract_loc}")
            except:
                pass
            
            os.mkdir(f"{self.char_extract_loc}/{wr}")

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

                try:
                # Save the cropped image
                    roi = cv2.resize(roi,(500,500))
                    cv2.imwrite(f'{self.char_extract_loc}/{wr}/{k}.jpg', roi)
                    k+=1
                except:
                    pass

            wr+=1

            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def recognize(self):
        for ltr in range(len(os.listdir(self.char_extract_loc))):
            data_dir = self.char_extract_loc + "/" +str(ltr+1)

            word = ""
            prev = 0

            for i in os.listdir(data_dir):
                z = data_dir + "/" + i

                image = Image.open(z)
                new_image = ImageOps.expand(image,border=70,fill='white')
                new_image = new_image.resize((1000,1000))
                new_image.save(z)

                img = cv2.imread(z)

                resize = tf.image.resize(img, (256,256))
                yhat = self.model.predict(np.expand_dims(resize/255, 0))
                pval = np.argmax(yhat)

                if pval not in [120,121,122,123]:
                    word += str(self.letters.iloc[pval,1])

                    if prev in [3014,3015,3016]:
                        word+=chr(prev)
                        prev=0

                else:
                    if pval==120 and ((len(word)-1)>=0):
                        if ord(word[len(word)-1])  == 3014: 
                            word = word[0:len(word)-1]
                            word+=chr(3018)
                        elif ord(word[len(word)-1])  == 3015: 
                            word = word[0:len(word)-1]
                            word+=chr(3019)
                        else:    
                            word+= chr(3006)

                    elif pval==121:
                        prev = 3014
                    elif pval==122:
                        prev = 3015
                    elif pval==123:
                        prev = 3016

            self.para+=word
            self.para+=" "

        print("Recognized Paragraph")
        print(self.para)

    def count_segregate(self):
      #character count
        for i in range(len(self.para)):
            if self.para[i] in self.uyir_list:
              self.uyir+=1
  
            elif self.para[i] in self.mei_list and ord(self.para[i+1])==3021:
                self.mei+=1
            
            elif self.para[i] in self.mei_list:
                self.uyir_mei+=1
            
            elif self.para[i]=="ஃ":
                self.ayutha+=1 
            
            else:
                pass
   
            if self.para[i] in self.vallinam_list and ord(self.para[i+1])==3021:
              self.vallinam+=1
  
            if self.para[i] in self.mellinam_list and ord(self.para[i+1])==3021: 
              self.mellinam+=1
  
            if self.para[i] in self.idayinam_list and ord(self.para[i+1])==3021:
              self.idayinam+=1
  
        self.total_characters = self.uyir+self.mei+self.uyir_mei+self.ayutha
        print("\nUyir Eluthukal : "+str(self.uyir))
        print("\nMei Eluthukal : "+str(self.mei))
        print("\nUyir-Mei Eluthukal : "+str(self.uyir_mei))
        print("\nAyutha Eluthu : "+str(self.ayutha))
        print("\nVallinam : "+str(self.vallinam))
        print("\nMellinam : "+str(self.mellinam))
        print("\nIdayinam : "+str(self.idayinam))
        print("\nTotal Length : "+str(self.total_characters))
    
    def transliterate(self):

        eng1 = ["a","aa","i","ii","u","uu","e","ee","ai","o","oo","au","aK"]
        eng2 = ["k","nk","ch","nc","t","Nn","th","n","p","m","y","r","l","v","z","L","R","N"] 

        for y in self.para:
            if y in self.uyir_list: 
                ind = self.uyir_list.index(y)
                self.eg+= eng1[ind]
                
            elif y in self.mei_list:
                ind = self.mei_list.index(y)
                self.eg+= eng2[ind]+"a"
                
            elif ord(y)==3021:
                self.eg = self.eg[0:len(self.eg)-1]
                continue

            elif ord(y) in self.asc:
                ind = self.asc.index(ord(y))
                self.eg = self.eg[0:len(self.eg)-1]
                self.eg+= eng1[ind]

            else:
                self.eg+= y

        print("\nTransliterad Paragraph")
        print(self.eg)
    
    def pronounce(self):
        tts = gt.gTTS(text=self.para, lang='ta')
        tts.save("Tamil-Audio.mp3")
        os.system("Tamil-Audio.mp3")