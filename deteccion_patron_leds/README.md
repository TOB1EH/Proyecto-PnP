# Estimador de Pose de LEDs Infrarojos

Este proyecto implementa un sistema de estimación de pose utilizando LEDs infrarrojos. El sistema detecta tres LEDs en una imagen de video en tiempo real y calcula su posición y orientación en el espacio 3D.

**Autor**
--------

* **Tobias Funes**
* **Correo electrónico:** tobiasfunes@hotmail.com.ar
* **GitHub:** TOB1EH

## Descripción del Código

El código está organizado en una clase principal `IRLedPoseEstimator` y una función `main()`. A continuación, se detalla cada componente:

### Clase IRLedPoseEstimator

#### Método `__init__`
- Carga los parámetros de calibración de la cámara desde archivos.
- Configura el patrón 3D de LEDs.
- Establece parámetros para la detección de LEDs y el filtrado temporal de la pose.

#### Método `detect_ir_leds`
- Convierte el frame a escala de grises.
- Aplica umbralización para detectar áreas brillantes.
- Utiliza `cv2.connectedComponentsWithStats` para encontrar blobs.
- Filtra los blobs por área para identificar LEDs válidos.
- Retorna las coordenadas de los LEDs si se detectan exactamente tres.

#### Método `estimate_pose`
- Utiliza `cv2.solvePnP` con los algoritmos SQPNP y P3P para estimar la pose.
- Aplica filtrado temporal para suavizar las estimaciones entre frames.
- Maneja excepciones y realiza intentos de respaldo si la estimación falla.

#### Método `draw_axes`
- Proyecta puntos 3D a coordenadas 2D en la imagen.
- Dibuja ejes X, Y, Z en colores rojo, verde y azul respectivamente.
- Calcula y muestra ángulos de Euler y distancia.

#### Método `rotationMatrixToEulerAngles`
- Convierte una matriz de rotación 3x3 a ángulos de Euler (roll, pitch, yaw).
- Maneja casos singulares (gimbal lock).

### Función main()

- Inicializa la cámara y crea ventanas para visualización.
- Ejecuta un bucle principal que:
  1. Captura frames de la cámara.
  2. Detecta LEDs usando `detect_ir_leds`.
  3. Estima la pose si se detectan 3 LEDs.
  4. Dibuja los ejes 3D si la estimación es exitosa.
  5. Muestra el frame original y una versión umbralizada.

## Requisitos

- Python 3.x
- OpenCV (cv2)
- NumPy
- Pickle

## Uso

1. Asegúrese de tener los archivos de calibración de la cámara en el directorio especificado.
2. Ejecute el script:
    ```
    python3 leds_pose_estimator.py
    ```
3. Apunte la cámara hacia el patrón de 3 LEDs.
4. Presione 'q' para salir del programa.

## Notas Técnicas

- El sistema está diseñado para un patrón específico de 3 LEDs en línea recta.
- Utiliza calibración previa de la cámara para mejorar la precisión.
- Implementa filtrado temporal para suavizar las estimaciones de pose.

## Limitaciones y Mejoras Futuras

- La detección puede ser sensible a las condiciones de iluminación.
- Podría extenderse para trabajar con diferentes patrones de LEDs.
- La optimización del rendimiento podría mejorar la velocidad de procesamiento en tiempo real.
