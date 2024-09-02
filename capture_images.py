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
        # cv2.imwrite('/home/tobias/Documentos/workSpace/Proyecto_pnp/solvepnp/calibracion/imagenes/img' + str(num) + '.png', img)
        cv2.imwrite('/home/tobias/Documentos/workSpace/Proyecto_pnp/solvepnp/calibracion/image_prueba.png', img)
        print("image saved!")
    
    # Salir con la tecla 'q'
    if key == ord('q'):
        break

# Liberar y destruir la camara
cap.release()
cv2.destroyAllWindows()