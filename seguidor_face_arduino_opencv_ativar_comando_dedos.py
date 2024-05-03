import cv2
from pyfirmata import Arduino, SERVO
from time import sleep
import mediapipe as mp

face = mp.solutions.face_detection
Face = face.FaceDetection()
mpDraw = mp.solutions.drawing_utils

port = 'COM3'
pinH = 10
pinV = 8
board = Arduino(port)

board.digital[pinH].mode = SERVO
board.digital[pinV].mode = SERVO

def rotateServo(pin, angle):
    # Garante que o valor de angle esteja dentro do intervalo permitido (0 a 255)
    angle = max(0, min(angle, 255))
    board.digital[pin].write(angle)
    sleep(0.015)

cap = cv2.VideoCapture(0)

positionX = 50
positionY = 70

rotateServo(pinH, positionX)
rotateServo(pinV, positionY)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

microfone_ativado = False

while True:
    ret, img = cap.read()
    if not ret:
        print("Erro ao ler a imagem da câmera.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Face.process(imgRGB)
    facesPoints = results.detections
    hO, wO, _ = img.shape

    cv2.line(img,(0,int(hO/2)),(wO,int(hO/2)),(0,255,0),2)
    cv2.line(img, (int(wO / 2), 0), (int(wO / 2), hO), (0, 255, 0), 2)

    if facesPoints:
        for id, detection in enumerate(facesPoints):
            bbox = detection.location_data.relative_bounding_box
            x,y,w,h = int(bbox.xmin*wO),int(bbox.ymin*hO),int(bbox.width*wO),int(bbox.height*hO)

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            # centro do rosto
            xx = int(x + (x + w)) // 2
            yy = int(y + (y + h)) // 2
            cv2.circle(img, (xx, yy), 15, (0, 255, 0), cv2.FILLED)

            ctX = int(wO / 2)
            ctY = int(hO / 2)

            cv2.circle(img, (ctX, ctY), 15, (255, 0, 0), cv2.FILLED)

            # movimento eixo X
            if xx < (ctX - 50):
                positionX += 1
                rotateServo(pinH, positionX)
            elif xx > (ctX + 50):
                positionX -= 1
                rotateServo(pinH, positionX)
            # movimento eixo Y
            if yy > (ctY + 50):
                positionY += 1
                rotateServo(pinV, positionY)
            elif yy < (ctY - 50):
                positionY -= 1
                rotateServo(pinV, positionY)

    # Detecção de mãos e dedos
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

            # Verifica se o dedo indicador e polegar estão juntos
            thumb_tip = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
            if thumb_tip.y < index_tip.y:
                if not microfone_ativado:
                    # Adicione aqui a lógica para ativar o microfone
                    print("Microfone ativado")
                    microfone_ativado = True
            else:
                microfone_ativado = False

    cv2.imshow('img', img)

    k = cv2.waitKey(1) & 0xff
    if k == 27:  # Pressione ESC para sair
        break
    elif k == ord('q'):  # Pressione 'q' para fechar a webcam
        cap.release()
        cv2.destroyAllWindows()
        break
