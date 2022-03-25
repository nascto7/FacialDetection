import cv2 #Biblioteca do OpenCV

BLUE_COLOR = (255, 0, 0) #Setando a cor azul por RGB
STROKE = 10 #Grossura da borda de detecção


cap = cv2.VideoCapture('dt.mp4') #Carregando a imagem
xml_path = 'haarcascade_frontalface_alt2.xml' #Atribuindo o modelo a variável.
clf = cv2.CascadeClassifier(xml_path) #Classificador de imagem

#Loop que mantem o código rodando até receber o comando para parar.
while(cap.isOpened()): 
    ret, frame = cap.read() #Lendo a variavel cap e retornando um binário e uma 'foto'. O loop é necessário pois estamos analisando um vídeo 
                            #e a leitura deve ser feita de forma contínua
  
    if not ret: 
        print('Ret está vazio. Favor verificar') #Se ret estiver vazio o loop é cortado.
        break  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Alterando os tons de cores do vídeo, pois o modelo só consegue ser usado em tons de cinza.
    faces = clf.detectMultiScale(gray) #Detecta objetos de tamanhos diferentes na imagem de entrada. Retornando uma lista de objetos detectados.
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), BLUE_COLOR, STROKE)#Detectar rosto
    
    cv2.imshow('frame', frame) #Exibe o vídeo em uma janela
  
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #Fecha a o vídeo se for pressionado a tecla 'q'
        break

cap.release() #Fecha o vídeo
cv2.destroyAllWindows() #Fecha todas as janelas abertas
