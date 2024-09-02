import cv2 as cv
import numpy as np
import glob
import pickle

### ENCONTRAR ESQUINAS DEL TABLERO DE AJEDREZ: PUNTOS DE OBJETO Y PUNTOS DE IMAGEN ###

# Definir la forma y tamaño del tablero de ajedrez (en este caso, 11 x 6)
nCols = 11 # nro de columnas - 1
nRows = 6 # nro de filas - 1

frameSize = (640,480) # tamanio de imagen capturada por la webcam

# Define los criterios de terminación para el refinamiento de esquinas (corners) en la función
# El criterio de terminación indica cuándo detener el algoritmo de minimización.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Definir los puntos 3D del tablero de ajedrez (en este caso, los puntos de la esquina superior izquierda y la esquina inferior derecha)
objp = np.zeros((nCols * nRows, 3), np.float32)  # Puntos 3D del tablero de ajedrez
objp[:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2) # crear un patrón de puntos 3D que representan la posición de los esquinas del patrón de calibración.


# Arrays para almacenar los puntos 3D y 2D de todos los objetos detectados
objpoints = []  
imgpoints = [] 

# Buscar todos los archivos con extensión.png en la carpeta de capturas
images = glob.glob('/home/tobias/Documentos/workSpace/Proyecto_pnp/solvepnp/capturas/img*.png')

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
pickle.dump(objp, open("/home/tobias/Documentos/workSpace/Proyecto_pnp/solvepnp/calibracion/objpoints.pkl", "wb"), protocol = 2)

# Guardar matriz de cámara (cameraMatrix) en archivo cameraMatrix.pkl
pickle.dump(cameraMatrix, open("/home/tobias/Documentos/workSpace/Proyecto_pnp/solvepnp/calibracion/cameraMatrix.pkl", "wb"), protocol = 2)

# Guardar coeficientes de distorsión (dist) en archivo distortion.pkl
pickle.dump(dist, open("/home/tobias/Documentos/workSpace/Proyecto_pnp/solvepnp/calibracion/distortion.pkl", "wb"), protocol = 2)

print("CALIBRACION FINALIZADA!")



""" 
### REMOVER DISTORCION DE UNA IMAGEN ###

# Load previously saved camera calibration result
img = cv.imread('/home/tobias/Documentos/workSpace/Proyecto_pnp/solvepnp/calibracion/cali.png') # cali5.png es la imagen de prueba
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('/home/tobias/Documentos/workSpace/Proyecto_pnp/solvepnp/calibracion/caliResult1.png', dst)

# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('/home/tobias/Documentos/workSpace/Proyecto_pnp/solvepnp/calibracion/caliResult2.png', dst)

# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error/len(objpoints)) )

 """




