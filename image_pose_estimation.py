import cv2 as cv
import numpy as np
import pickle

nCols = 11
nRows = 6

# El criterio de terminación indica cuándo detener el algoritmo de minimización.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Cargar la imagen que deseas procesar
img = cv.imread('/home/tobias/Documentos/workSpace/Proyecto_pnp/solvepnp/test_image.png')


with open('/home/tobias/Documentos/workSpace/Proyecto_pnp/solvepnp/calibracion/objpoints.pkl', 'rb') as f:
    objp = pickle.load(f)

with open('/home/tobias/Documentos/workSpace/Proyecto_pnp/solvepnp/calibracion/cameraMatrix.pkl', 'rb') as f:
    cameraMatrix = pickle.load(f)

with open('/home/tobias/Documentos/workSpace/Proyecto_pnp/solvepnp/calibracion/distortion.pkl', 'rb') as f:
    dist = pickle.load(f)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, (nCols, nRows), None)

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


    def tupleOfInts(array):
        return tuple(int(x) for x in array) 

    corner = tupleOfInts(cornersRefined[0].ravel())
    img = cv.line(img, corner, tupleOfInts(imgpts[0].ravel()), (255,0,0), 2)
    img = cv.line(img, corner, tupleOfInts(imgpts[1].ravel()), (0,255,0), 2)
    img = cv.line(img, corner, tupleOfInts(imgpts[2].ravel()), (0,0,255), 2)

    # Muestra la imagen con los ejes de la cámara
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()