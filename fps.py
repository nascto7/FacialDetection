import cv2 #Biblioteca do OpenCV

BLUE_COLOR = (255, 0, 0)
STROKE = 10


cap = cv2.VideoCapture('dt.mp4')
xml_path = 'haarcascade_frontalface_alt2.xml'
clf = cv2.CascadeClassifier(xml_path) #Classificador de imagem

while(cap.isOpened()): 
    ret, frame = cap.read() 
  
    if not ret: 
        print('Ret está vazio. Favor verificar')
        break  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(gray) 
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), BLUE_COLOR, STROKE)
    
    cv2.imshow('frame', frame) 
  
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release() 
cv2.destroyAllWindows() 