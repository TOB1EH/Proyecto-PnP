# Este script utiliza OpenCV para capturar una imagen desde la webcam 
# y obtiene sus dimensiones (ancho y alto), que luego imprime en la consola.

# 2024 Tobias Funes (tobiasfunes@hotmail.com.ar)

import cv2

# Inicializa la webcam
cap = cv2.VideoCapture(0)

# Captura una imagen
ret, frame = cap.read()

# Obtiene las dimensiones de la imagen
height, width, _ = frame.shape

# Imprime las dimensiones de la imagen
print(f"El tamaño de la imagen es: {width}x{height}")

# Cierra la webcam
cap.release()