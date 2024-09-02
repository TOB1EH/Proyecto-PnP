# Este archivo contiene constantes utilizadas para la captura y calibración de imágenes de un tablero de ajedrez. 
# Define las dimensiones del tablero, el tamaño de la imagen capturada, criterios de terminación para el refinamiento 
# de esquinas, y las rutas de los directorios para las capturas y calibración.

# 2024 Tobias Funes (tobiasfunes@hotmail.com.ar)

import cv2 as cv

### CONSTANTES ###

# Definir la forma y tamaño del tablero de ajedrez (en este caso, 11 x 6)
nCols = 11 # nro de columnas - 1 (columnas internas)
nRows = 6 # nro de filas - 1 (filas internas)

# Tamanio de imagen capturada por la webcam
frameSize = (640,480) # En este caso, 640x480 

# Define los criterios de terminación para el refinamiento de esquinas (corners) en la función
# El criterio de terminación indica cuándo detener el algoritmo de minimización.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


### RUTAS DE ARCHIVOS ###

CAPTURES_DIR = 'capturas/' # Rutas para la carpeta de capturas
CALIBRATION_DIR = 'calibracion/' # Rutas para la carpeta de calibracion

