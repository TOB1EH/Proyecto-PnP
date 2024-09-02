# Este script realiza la estimación de pose en tiempo real utilizando un tablero de ajedrez 
# para la calibración de la cámara. Carga los parámetros de calibración desde archivos, 
# captura video en tiempo real, detecta los puntos 2D del tablero de ajedrez, refina estos 
# puntos, estima la pose de la cámara y dibuja los ejes de la cámara en la imagen capturada.

# 2024 Tobias Funes (tobiasfunes@hotmail.com.ar)

import cv2 as cv
import numpy as np
import pickle
from constants import nCols, nRows, criteria, CALIBRATION_DIR

### CARGAR CALIBRACION ###
with open(CALIBRATION_DIR + 'objpoints.pkl', 'rb') as f:
    objp = pickle.load(f)

with open(CALIBRATION_DIR + 'cameraMatrix.pkl', 'rb') as f:
    cameraMatrix = pickle.load(f)

with open(CALIBRATION_DIR + 'distortion.pkl', 'rb') as f:
    dist = pickle.load(f)


# Captura de video en tiempo real
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detectar puntos 2D del tablero de ajedrez
    ret, corners = cv.findChessboardCorners(gray, (nCols, nRows), None)

    if ret == True:
        # Refinar puntos 2D detectados
        cornersRefined = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        # Estimar pose de la cámara utilizando solvePnP
        ret, rvec, tvec = cv.solvePnP(objp, cornersRefined, cameraMatrix, dist)

        # Dibujar ejes de la cámara en la imagen
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]])
        imgpts, jac = cv.projectPoints(axis, rvec, tvec, cameraMatrix, dist)

        # Dibujar ejes de la cámara
        def tupleOfInts(array):
            return tuple(int(x) for x in array) 

        corner = tupleOfInts(cornersRefined[0].ravel())
        frame = cv.line(frame, corner, tupleOfInts(imgpts[0].ravel()), (255,0,0), 2)
        frame = cv.line(frame, corner, tupleOfInts(imgpts[1].ravel()), (0,255,0), 2)
        frame = cv.line(frame, corner, tupleOfInts(imgpts[2].ravel()), (0,0,255), 2)

    # Mostrar la imagen en tiempo real
    cv.imshow('img', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()