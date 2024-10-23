# Este script realiza la estimación de la pose de una cámara utilizando una imagen de un tablero de ajedrez. 
# Carga los parámetros de calibración de la cámara y los puntos del objeto desde archivos, 
# detecta los bordes del tablero en la imagen, refina los puntos detectados, 
# estima la rotación y la traslación de la cámara, y finalmente dibuja los ejes de la cámara en la imagen resultante.

# 2024 Tobias Funes (tobiasfunes@hotmail.com.ar)

import cv2 as cv
import numpy as np
import pickle
from patron_ajedrez.constants import nCols, nRows, criteria, CALIBRATION_DIR

# Cargar la imagen que deseas procesar
img = cv.imread('test_image.png')

### CARGAR CALIBRACION ###

with open(CALIBRATION_DIR + 'objpoints.pkl', 'rb') as f:
    objp = pickle.load(f)

with open(CALIBRATION_DIR + 'cameraMatrix.pkl', 'rb') as f:
    cameraMatrix = pickle.load(f)

with open(CALIBRATION_DIR + 'distortion.pkl', 'rb') as f:
    dist = pickle.load(f)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, (nCols, nRows), None)


### ESTIMAR POSE ###
if ret == True:
    # Refina los puntos 2D detectados
    cornersRefined = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    # Estima la pose de la cámara utilizando solvePnP
    ret, rvec, tvec = cv.solvePnP(objp, cornersRefined, cameraMatrix, dist)

    # Muestra la pose estimada
    print("POSE ESTIMADA:")
    print("Rotación:\n", rvec)
    print("Traslación:\n", tvec)

    # Dibuja los ejes de la cámara en la imagen
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]])
    imgpts, jac = cv.projectPoints(axis, rvec, tvec, cameraMatrix, dist)

    # convertir la lista de enteros en una tupla
    def tupleOfInts(array):
        return tuple(int(x) for x in array)

    corner = tupleOfInts(cornersRefined[0].ravel())
    # El método ravel() se aplica al elemento seleccionado, lo que 
    # devuelve un array plano (1D) de los elementos del array original.

    # Dibujar las lineas en la imagen
    img = cv.line(img, corner, tupleOfInts(imgpts[0].ravel()), (255,0,0), 2)
    img = cv.line(img, corner, tupleOfInts(imgpts[1].ravel()), (0,255,0), 2)
    img = cv.line(img, corner, tupleOfInts(imgpts[2].ravel()), (0,0,255), 2)

    # Muestra la imagen con los ejes de la cámara
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()