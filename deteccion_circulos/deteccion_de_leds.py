import cv2
import numpy as np

# Abrir la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer una imagen de la cámara
    ret, frame = cap.read()

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral para detectar las luces LEDs
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Encontrar los círculos que representan los LEDs
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=5, maxRadius=50)

    # Recorrer los círculos y dibujar un círculo alrededor de cada uno
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Mostrar la imagen
    cv2.imshow('Frame', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara
cap.release()
cv2.destroyAllWindows()