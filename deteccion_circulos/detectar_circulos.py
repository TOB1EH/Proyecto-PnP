import cv2
import numpy as np

# Abrir la webcam
cap = cv2.VideoCapture(0)

while True:
    # Leer una imagen de la webcam
    ret, frame = cap.read()

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro de Gauss
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar un umbral de intensidad
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Detectar círculos utilizando el algoritmo de RANSAC
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los círculos detectados
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        if aspect_ratio > 0.5 and aspect_ratio < 2:
            cv2.circle(frame, (x + w // 2, y + h // 2), w // 2, (0, 255, 0), 2)

    # Mostrar la imagen
    cv2.imshow('Frame', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la webcam
cap.release()
cv2.destroyAllWindows()