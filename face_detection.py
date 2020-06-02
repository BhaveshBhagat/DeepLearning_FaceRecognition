import numpy as np
import cv2

''' defining the fuction for extract face from the image frame '''
def face_extract(img):
    
    #faseCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    faces = faseCascade.detectMultiScale(img ,1.1 ,2)

    if faces is ():
        return None
        
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50 , x:x+w+50]
    
    return cropped_face


'''Creating object of opencv CascadeClassifier for face detection '''
faseCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

''' Taking image for face detection '''

img = cv2.imread('C:/Users/Bhagat/Documents/Python/DeepLearning/photos/tt1.jpeg')

''' converting image into gray scale '''
gray_image = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

''' detecting the faces in the image '''
faces = faseCascade.detectMultiScale(gray_image , 1.1 ,4)

''' finally creating a boundry box on image faces '''    
for (x,y,w,h) in faces:
        
    cv2.rectangle(img , (x,y),(x+w,y+h),(0,255,0),2)

''' then show the image '''
cv2.imshow('image' , img)



''' Next We try to detect faces in video frames or using Webcam '''

count = 0

while True:
    ''' Using webcam for capturing the video frame as image '''
    cap = cv2.VideoCapture(0)

    op , img = cap.read()

    gray_image = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

    faces = faseCascade.detectMultiScale(gray_image , 1.1 ,2)
    
    for (x,y,w,h) in faces:
        
         cv2.rectangle(img , (x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('image' , img)

    if cv2.waitKey(1) == 13: # 13 is the Enter key
        break
    
    '''Below Code use for extracting the faces and save in the given directory '''
    '''
    ret, img = cap.read()    

    if face_extract(img) is not None:
         count +=1
         face = cv2.resize(face_extract(img), (400,400))

         file_name_path = 'C:/Users/Bhagat/Documents/Python/DeepLearning/DataSet_image/1/'+str(count)+'.jpg'
         cv2.imwrite(file_name_path, face)

         cv2.putText(face , str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
         cv2.imshow('Face cropper', face)

    else:
        print('Not found')
        pass
    
    if cv2.waitKey(1) ==13 or count==200:
         # 13 is the Enter key
         break
   '''


''' And finally we are releasing and destroying all the windows we uses '''
cap.release()
cv2.destroyAllWindows()






