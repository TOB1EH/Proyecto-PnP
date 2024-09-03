Estimación de Pose de Cámara en Tiempo Real
=============================================

Este proyecto utiliza la biblioteca OpenCV para estimar la pose de una cámara en tiempo real utilizando un patrón de tablero de ajedrez. El proyecto se basa en la implementación de la función SolvePnP en un contexto de tiempo real y recursos de hardware definidos (Sistemas Linux, utilizando Python).

**Autor**
--------

* **Tobias Funes**
* **Correo electrónico:** tobiasfunes@hotmail.com.ar
* **GitHub:** TOB1EH

Capturas del Proyecto
-------------------------

[Agregar imágenes  del proyecto]

Requisitos Previos
---------------------

### Instalar OpenCV para Python

Para instalar OpenCV para Python, puedes seguir los siguientes pasos:

1. Instalar pip: `sudo apt-get install python3-pip` (para sistemas Linux)
2. Instalar OpenCV: `pip3 install opencv-python`

Estructura del Proyecto
-------------------------

El proyecto se compone de los siguientes archivos:

* [capture_images.py](#capture_imagespy)
* [constants.py](#constantspy)
* [calibrate_camera.py](#calibrate_camerapy)
* [image_pose_estimation.py](#image_pose_estimationpy)
* [real_time_pose_estimation.py](#real_time_pose_estimationpy)
* [get_image_dimensions.py](#get_image_dimensionspy)

Descripción de los Archivos
-----------------------------

### capture_images.py

Este script utiliza OpenCV para capturar imágenes desde una webcam. Permite al usuario capturar múltiples imágenes presionando la tecla 'c' y salir del programa presionando la tecla 'q'. Las imágenes se guardan en una ruta específica.


### constants.py

Este archivo contiene constantes utilizadas para la captura y calibración de imágenes de un tablero de ajedrez. Define las dimensiones del tablero, el tamaño de la imagen capturada, criterios de terminación para el refinamiento de esquinas, y las rutas de los directorios para las capturas y calibración.

### calibrate_camera.py

Este script realiza la calibración de una cámara utilizando un patrón de tablero de ajedrez. Primero, detecta las esquinas del tablero en varias imágenes y almacena los puntos 3D y 2D correspondientes. Luego, utiliza estos puntos para calcular la matriz de la cámara y los coeficientes de distorsión, que se guardan en archivos.

### image_pose_estimation.py

Este script realiza la estimación de la pose de una cámara utilizando una imagen de un tablero de ajedrez. Carga los parámetros de calibración de la cámara y los puntos del objeto desde archivos, detecta los bordes del tablero en la imagen, refina los puntos detectados, estima la rotación y la traslación de la cámara, y finalmente dibuja los ejes de la cámara en la imagen resultante.


### real_time_pose_estimation.py

Este script realiza la estimación de pose en tiempo real utilizando un tablero de ajedrez para la calibración de la cámara. Carga los parámetros de calibración desde archivos, captura video en tiempo real, detecta los puntos 2D del tablero de ajedrez, refina estos puntos, estima la pose de la cámara y dibuja los ejes de la cámara en la imagen capturada.

