import cv2
face = cv2.CascadeClassifier(cv2.data)

face_cassifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image = cv2.imread('example2.jpg')
img_color =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector = face_cassifier.detectMultiScale(img_color, scaleFactor=1.1, minNeighbors=5, minSize=(30,30) ,maxSize=(200,200))

for (i, j, k, x) in detector:
    cv2.rectangle(image,(i,j),(i+k,j+x), (0,255,0), 2)
cv2.imshow('image', image)