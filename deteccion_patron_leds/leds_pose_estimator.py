import cv2  # Biblioteca OpenCV para procesamiento de imágenes y visión por computadora
import numpy as np  # Biblioteca para operaciones numéricas y matriciales
import pickle  # Biblioteca para cargar archivos de calibración serializados

class IRLedPoseEstimator:
    def __init__(self):
        """
        Constructor de la clase. Inicializa todos los parámetros necesarios para la detección
        y estimación de pose de LEDs.
        """
        
        # Cargar los parámetros de calibración de la cámara desde archivos
        CALIBRATION_DIR = '../deteccion_patron_ajedrez/calibracion/'
        # La calibración se obtuvo con el tablero de ajedrez, esta misma es válida porque 
        # lo que calibró fue la cámara en sí (sus parámetros intrínsecos), no el patrón. 
        # Entonces esta calibración es válida para cualquier objeto que se quiera detectar 
        # con la misma cámara (webcam de mi laptop).

        # camera_matrix: matriz 3x3 que contiene los parámetros intrínsecos de la cámara
        # (distancia focal y punto principal)
        with open(CALIBRATION_DIR + 'cameraMatrix.pkl', 'rb') as f:
            self.camera_matrix = pickle.load(f)
        
        # dist_coeffs: coeficientes de distorsión de la cámara
        # (corrección de distorsión radial y tangencial)
        with open(CALIBRATION_DIR + 'distortion.pkl', 'rb') as f:
            self.dist_coeffs = pickle.load(f)

        # Configuración del patrón 3D de LEDs
        self.led_spacing = 4.0  # Distancia entre LEDs en centímetros
        
        # Definición de las coordenadas 3D de los LEDs en el espacio
        # El patrón es una línea recta horizontal con tres puntos:
        # LED izquierdo (-4,0,0), LED central (0,0,0), LED derecho (4,0,0)
        self.led_pattern_3d = np.array([
            [-self.led_spacing, 0, 0],  # LED izquierdo
            [0, 0, 0],                  # LED central
            [self.led_spacing, 0, 0]    # LED derecho
        ], dtype=np.float32)

        # Parámetros para la detección de LEDs en la imagen
        self.brightness_threshold = 250     # Umbral de brillo (0-255) para detectar LEDs
        self.min_blob_area = 20             # Área mínima en píxeles para considerar un blob como LED
        self.max_blob_area = 300            # Área máxima en píxeles para considerar un blob como LED

        # Variables para el filtrado temporal de la pose
        self.last_rvec = None          # Último vector de rotación calculado
        self.last_tvec = None          # Último vector de traslación calculado
        self.pose_filter_alpha = 0.5   # Factor de suavizado (0-1):
                                        # 0 = solo usar pose anterior
                                        # 1 = solo usar pose actual
                                        # 0.5 = promedio entre anterior y actual

    def detect_ir_leds(self, frame):
        """
        Detecta los LEDs infrarojos (IR) en el frame de la cámara.
        
        Args:
            frame: Imagen BGR capturada por la cámara
            
        Returns:
            tuple: (bool, np.array) - (éxito de detección, coordenadas de los LEDs)
        """

        # Convertir el frame a escala de grises para procesamiento
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar umbralización: píxeles > threshold se vuelven blancos (255)
        # píxeles <= threshold se vuelven negros (0)
        _, thresh = cv2.threshold(gray, self.brightness_threshold, 255, cv2.THRESH_BINARY)

        # Encontrar todos los componentes conectados (blobs) en la imagen umbralizada
        # num_labels: número total de blobs encontrados
        # labels: matriz del tamaño de la imagen donde cada píxel tiene el ID de su blob
        # stats: matriz con estadísticas de cada blob (área, perímetro, etc.)
        # centroids: coordenadas (x,y) del centro de cada blob
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

        # Filtrar los blobs por área para identificar LEDs válidos
        valid_centroids = [] # lista para coordenadas validas
        for i in range(1, num_labels):  # Empezar desde 1 para saltar el fondo (ID 0)
            area = stats[i, cv2.CC_STAT_AREA]  # Obtener área del blob actual
            if self.min_blob_area < area < self.max_blob_area:
                valid_centroids.append(centroids[i])

        # Verificar si encontramos exactamente 3 LEDs
        if len(valid_centroids) == 3:
            # Ordenar los LEDs de izquierda a derecha según su coordenada X
            valid_centroids.sort(key=lambda x: x[0])
            print("Se detectaron exactamente 3 LEDs")
            # Retorna 'found_leds' (True porque si encontro 3 leds) y 'led_centers' (arreglo 
            # de tipo float32 con las coordenadas del centro de los leds)
            return True, np.array(valid_centroids, dtype=np.float32)
        else:
            print(f"Se detectaron {len(valid_centroids)} LEDs, necesita exactamente 3")
            # Retorna 'found_leds' (False porque no encontro 3 leds) y 'led_centers' (None 
            # porque no hay exactamente 3 leds)
            return False, None

    def estimate_pose(self, led_centers):
        """
        Estima la pose (rotación y traslación) del patrón de LEDs.
        
        Args:
            led_centers: Array de coordenadas 2D de los centros de los LEDs (lista de tuplas)
            
        Returns:
            tuple: (bool, np.array, np.array) - (éxito, vector rotación, vector traslación)
        """

        try:
            # Intentar primero con SQPNP (Scaled Quaternion-based Perspective-n-Point)
            # Este método es más estable para conjuntos pequeños de puntos (Mi caso 3 puntos leds)
            success, rvec, tvec = cv2.solvePnP(
                self.led_pattern_3d,        # Coordenadas 3D de los LEDs en el mundo real
                led_centers,                # Coordenadas 2D de los LEDs en la imagen
                self.camera_matrix,         # Matriz de parámetros intrínsecos de la cámara (Archivo de calibracion)
                self.dist_coeffs,           # Coeficientes de distorsión de la cámara (Archivo de calibracion)
                flags=cv2.SOLVEPNP_SQPNP    # Método de resolución (SQPNP)
            )

            if not success:
                # Si SQPNP falla, intentar con P3P (Perspective-3-Point)
                success, rvec, tvec = cv2.solvePnP(
                    self.led_pattern_3d,        # Coordenadas 3D de los LEDs en el mundo real
                    led_centers,                # Coordenadas 2D de los LEDs en la imagen
                    self.camera_matrix,         # Matriz de parámetros intrínsecos de la cámara (Archivo de calibracion)
                    self.dist_coeffs,           # Coeficientes de distorsión de la cámara (Archivo de calibracion)
                    flags=cv2.SOLVEPNP_P3P      # Método de resolución (P3P)
                )

            if success:
                # Aplicar filtrado temporal para suavizar el movimiento
                if self.last_rvec is not None and self.last_tvec is not None:
                    # Interpolación lineal entre la pose anterior y la actual
                    rvec = self.last_rvec * (1 - self.pose_filter_alpha) + rvec * self.pose_filter_alpha
                    tvec = self.last_tvec * (1 - self.pose_filter_alpha) + tvec * self.pose_filter_alpha

                # Guardar la pose actual para el próximo frame
                self.last_rvec = rvec.copy()
                self.last_tvec = tvec.copy()
                print("Pose estimada exitosamente")
                # Retorna success (indica si solvePnP se ejecuto con exito), rvec (vector de rotacion) y tvec (vector de traslacion)
                return True, rvec, tvec
            else:
                print("No se pudo estimar la pose :(")
                # Retorna success (indica si solvePnP se ejecuto con exito), rvec (vector de rotacion) y tvec (vector de traslacion)
                return False, None, None

        except cv2.error as e:
            # Si ocurre una excepcion/error lo muestra:
            print(f"Error en estimación de pose: {str(e)}")
            # Intento final con P3P si todo lo demás falla
            try:
                success, rvec, tvec = cv2.solvePnP(
                    self.led_pattern_3d,
                    led_centers,
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_P3P
                )
                if success:
                    print("Pose estimada exitosamente con P3P como respaldo!")
                    # Retorna success (indica si solvePnP se ejecuto con exito), rvec (vector de rotacion) y tvec (vector de traslacion)
                    return True, rvec, tvec
            except cv2.error as e2:
                # Si tambien falla muestra el error:
                print(f"Error también en P3P: {str(e2)}")
            # Retorna success (indica si solvePnP se ejecuto con exito), rvec (vector de rotacion) y tvec (vector de traslacion)
            return False, None, None

    def draw_axes(self, frame, rvec, tvec):
        """
        Dibuja los ejes 3D y la información de pose sobre el frame.
        
        Args:
            frame: Imagen donde dibujar
            rvec: Vector de rotación
            tvec: Vector de traslación
        """
        try:
            # Definir puntos para dibujar los ejes 3D
            axis_length = self.led_spacing * 2  # Longitud de los ejes en centímetros
            
            # Definir los puntos que forman los ejes:
            # origen (0,0,0), fin eje X, fin eje Y, fin eje Z
            axis_points = np.float32([[0, 0, 0],
                                    [axis_length, 0, 0],  # Eje X
                                    [0, axis_length, 0],  # Eje Y
                                    [0, 0, axis_length]]) # Eje Z

            # Proyectar los puntos 3D a coordenadas 2D en la imagen
            imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            """
            imgpts: coordenadas 2D de los puntos 3D proyectados en la imagen (en pixeles)
            _ : vector jacobiano (no se usa en este caso)

            Args:
                axis_points: puntos 3D que forman los ejes de coordenadas cartesianas (x, y, z)
                rvec: vector de rotación
                tvec: vector de traslación
                camera_matrix: matriz de la cámara
                dist_coeffs: distorsión de la cámara
            """

            # Dibujar los ejes como líneas de colores
            origin = tuple(map(int, imgpts[0].ravel()))  # Punto de origen desde el cual se dibujaran los ejes
                                                         # Se obtiene del primer punto proyectado de 'imgpts'
                                                         # Representa el origen de coordenadas en la imagen.
                                                         # La función ravel() convierte el array en un vector 
                                                         # plano, y map(int, ...) convierte los valores a enteros, 
                                                         # ya que las coordenadas de píxeles deben ser enteras.
            
            # 'frame' es la imagen sobre la que se esta dibujando los ejes
            # cv2.line() se utiliza para dibujar una linea en la imagen 
            frame = cv2.line(frame, origin, tuple(map(int, imgpts[1].ravel())), (0,0,255), 3)   # X = Rojo
            frame = cv2.line(frame, origin, tuple(map(int, imgpts[2].ravel())), (0,255,0), 3)   # Y = Verde
            frame = cv2.line(frame, origin, tuple(map(int, imgpts[3].ravel())), (255,0,0), 3)   # Z = Azul

            # Calcular y mostrar ángulos de Euler y distancia
            rot_matrix, _ = cv2.Rodrigues(rvec)  # Convertir vector de rotación a matriz
            angles = self.rotationMatrixToEulerAngles(rot_matrix)  # Obtener ángulos de Euler
            
            # Mostrar información en la imagen
            text_angles = f"ANGULOS: Roll:{angles[0]:.1f} Pitch:{angles[1]:.1f} Yaw:{angles[2]:.1f}" # Rotaciones en el espacio tridimensional
            text_dist = f"DISTANCIA: {tvec[2][0]:.1f} cm"  # Distancia en Z
            cv2.putText(frame, text_angles, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, text_dist, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            print("Ejes dibujados correctamente")
            
        except Exception as e:
            print(f"Error al dibujar ejes: {str(e)}")

    def rotationMatrixToEulerAngles(self, R):
        """
        Convierte una matriz de rotación 3x3 a ángulos de Euler (en grados).
        
        Args:
            R: Matriz de rotación 3x3
            
        Returns:
            np.array: [roll, pitch, yaw] en grados
        """
        
        # Calcular el ángulo pitch (y)
        sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6  # Verificar si hay singularidad (gimbal lock)

        if not singular:
            # Caso normal: calcular los tres ángulos
            x = np.arctan2(R[2,1], R[2,2])     # Roll
            y = np.arctan2(-R[2,0], sy)        # Pitch
            z = np.arctan2(R[1,0], R[0,0])     # Yaw
        else:
            # Caso singular: pitch cerca de 90 grados
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0

        # Convertir radianes a grados
        return np.array([x, y, z]) * 180.0 / np.pi

def main():
    """
    Función principal que ejecuta el programa de detección y estimación de pose.
    """
    
    # Crear instancia del estimador
    estimator = IRLedPoseEstimator()

    # Inicializar la cámara (0 = cámara predeterminada)
    cap = cv2.VideoCapture(0)

    # Crear ventanas para visualización
    cv2.namedWindow('Original')   # Muestra imagen original con ejes
    cv2.namedWindow('Imagen Umbralizada')  # Muestra imagen umbralizada (Threshold)

    # Bucle principal
    while True:
        # Capturar frame de la cámara
        ret, frame = cap.read()
        if not ret: # si la lectura del frame no fue exitosa
            break

        # Detectar los LEDs en el frame actual
        found_leds, led_centers = estimator.detect_ir_leds(frame)

        # Crear imagen umbralizada para visualización
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar umbral a la imagen (frame)
        _, thresh = cv2.threshold(gray, estimator.brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Convierte la imagen (frame) umbralizada (que está en escala de grises) de nuevo a un formato de color BGR
        thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # si encuentra LEDs en el frame
        if found_leds: 
            # Dibujar círculos amarillos en las posiciones de los LEDs detectados
            for center in led_centers:
                cv2.circle(frame, tuple(map(int, center)), 5, (0, 255, 255), -1)
                cv2.circle(thresh_rgb, tuple(map(int, center)), 5, (0, 255, 255), -1)

            # Estimar la pose usando los centros de los LEDs
            success, rvec, tvec = estimator.estimate_pose(led_centers)

            if success:
                # Si la estimación fue exitosa, dibujar los ejes 3D
                estimator.draw_axes(frame, rvec, tvec)

        # Mostrar las imágenes
        cv2.imshow('Original', frame)
        cv2.imshow('Imagen Umbralizada', thresh_rgb)

        # Verificar si se presionó 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

# Punto de entrada del programa
if __name__ == "__main__":
    main()