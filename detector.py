import cv2

#Load the face classifier 
face_cassifier = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

#Start the video capture whith the camera
cap = cv2.VideoCapture(0)

while True:
    #Capture a frame of the video capture
    ret, frame = cap.read()

    if not ret:
        print("Error al capturar el fotograma")
        break

    #Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces in the frame
    faces = face_cassifier.detectMultiScale(gray, 1.1, 4)

    #Draw rectangles arround the faces that are detected
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (231, 74, 74), 3)

    #Show the frame whit the detected faces
    cv2.imshow('Video con Detección de Rostros', frame)

    #Press "c" to exit 
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

#Close all the windows
cap.release()
cv2.destroyAllWindows()
