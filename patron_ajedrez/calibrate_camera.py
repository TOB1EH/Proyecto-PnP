# Este script realiza la calibración de una cámara utilizando un patrón de tablero de ajedrez. 
# Primero, detecta las esquinas del tablero en varias imágenes y almacena los puntos 3D y 2D correspondientes. 
# Luego, utiliza estos puntos para calcular la matriz de la cámara y los coeficientes de distorsión, 
# que se guardan en archivos. Finalmente, se incluye un método para eliminar la distorsión de una imagen 
# usando los parámetros obtenidos durante la calibración.

# 2024 Tobias Funes (tobiasfunes@hotmail.com.ar)

import cv2 as cv
import numpy as np
import glob
import pickle
from patron_ajedrez.constants import nCols, nRows, criteria, frameSize,CAPTURES_DIR, CALIBRATION_DIR

### ENCONTRAR ESQUINAS DEL TABLERO DE AJEDREZ: PUNTOS DE OBJETO Y PUNTOS DE IMAGEN ###

# Definir los puntos 3D del tablero de ajedrez (en este caso, los puntos de la esquina superior izquierda y la esquina inferior derecha)
objp = np.zeros((nCols * nRows, 3), np.float32)  # Puntos 3D del tablero de ajedrez
objp[:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2) # crear un patrón de puntos 3D que representan la posición de los esquinas del patrón de calibración.

# Arrays para almacenar los puntos 3D y 2D de todos los objetos detectados
objpoints = []  
imgpoints = [] 

# Buscar todos los archivos con extensión.png en la carpeta de capturas
images = glob.glob(CAPTURES_DIR + 'img*.png')

## Recorrer todas las imagenes ##
for image in images:

    img = cv.imread(image) # lee una imagen desde un archivo y la carga en memoria como una matriz de píxeles
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convierte la imagen a escala de grises

    # detectar las esquinas en el patron de calibracion en una imagen
    ret, corners = cv.findChessboardCorners(gray, (nCols, nRows), None) 

    if ret == True:

        # refinar la localización de esquinas en una imagen
        cornersRefined = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) 
        
        objpoints.append(objp) 
        imgpoints.append(cornersRefined)

        # Dibujar y mostrar las esquinas
        cv.drawChessboardCorners(img, (nCols, nRows), cornersRefined, ret)
        cv.imshow('img', img)
        cv.waitKey(100)

cv.destroyAllWindows() # cerrar ventanas abiertas por OpenCV

### CALIBRACION ###

# Calibrar la cámara
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Guardar puntos de objeto (objp) en archivo objpoints.pkl
pickle.dump(objp, open(CALIBRATION_DIR + "objpoints.pkl", "wb"), protocol = 2)

# Guardar matriz de cámara (cameraMatrix) en archivo cameraMatrix.pkl
pickle.dump(cameraMatrix, open(CALIBRATION_DIR + "cameraMatrix.pkl", "wb"), protocol = 2)

# Guardar coeficientes de distorsión (dist) en archivo distortion.pkl
pickle.dump(dist, open(CALIBRATION_DIR + "distortion.pkl", "wb"), protocol = 2)

print("CALIBRACION FINALIZADA!")