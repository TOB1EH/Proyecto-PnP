import cv2 as cv
import numpy as np
import glob
import pickle
from patron_ajedrez.constants import nCols, nRows, criteria, frameSize, CAPTURES_DIR, CALIBRATION_DIR

### ENCONTRAR ESQUINAS DEL CUBO: PUNTOS DE OBJETO Y PUNTOS DE IMAGEN ###

# Definir los puntos 3D del cubo
objp = np.zeros((8, 3), np.float32)
objp[:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2) # crear un patrón de puntos 3D que representan la posición de los esquinas del patrón de calibración.

# Arrays para almacenar los puntos 3D y 2D de todos los objetos detectados
objpoints = []
imgpoints = []

# Buscar todos los archivos con extensión.png en la carpeta de capturas
images = glob.glob('capturas_leds/' + 'leds*.png')

## Recorrer todas las imagenes
for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detectar las esquinas en el patron de calibracion en una imagen
    ret, corners = cv.findChessboardCorners(gray, (2, 2), None)

    if ret == True:
        # Refinar la localización de esquinas en una imagen
        cornersRefined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(cornersRefined)

        # Dibujar y mostrar las esquinas
        cv.drawChessboardCorners(img, (2, 2), cornersRefined, ret)
        cv.imshow('img', img)
        cv.waitKey(100)

cv.destroyAllWindows()

### CALIBRACION ###

# Calibrar la cámara
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Guardar puntos de objeto (objp) en archivo objpoints.pkl
pickle.dump(objp, open(CALIBRATION_DIR + "objpoints_cubo.pkl", "wb"), protocol=2)

# Guardar matriz de cámara (cameraMatrix) en archivo cameraMatrix.pkl
pickle.dump(cameraMatrix, open(CALIBRATION_DIR + "cameraMatrix_cubo.pkl", "wb"), protocol=2)

# Guardar coeficientes de distorsión (dist) en archivo distortion.pkl
pickle.dump(dist, open(CALIBRATION_DIR + "distortion_cubo.pkl", "wb"), protocol=2)

print("CALIBRACION FINALIZADA!")