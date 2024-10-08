# Este script utiliza OpenCV para capturar imágenes desde una webcam. 
# Permite al usuario capturar múltiples imágenes presionando la tecla 
# 'c' y salir del programa presionando la tecla 'q'. Las imágenes se 
# guardan en una ruta específica.

# 2024 Tobias Funes (tobiasfunes@hotmail.com.ar)

import cv2

# Inicializar la webcam
cap = cv2.VideoCapture(0)

num = 0 

while cap.isOpened():
    #Capturar un frame
    ret, img = cap.read()

    # Mostrar el frame
    cv2.imshow('frame', img)
    
    # Preguntar al usuario si desea capturar la imagen (usar la tecla c)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        cv2.imwrite('capturas/img' + str(num) + '.png', img) # Capturar muchas imagenens
        # cv2.imwrite('image_prueba.png', img) # Solo para capturar una imagen
        print("image saved!")
    
    # Salir con la tecla 'q'
    if key == ord('q'):
        break

# Liberar y destruir la camara
cap.release()
cv2.destroyAllWindows()