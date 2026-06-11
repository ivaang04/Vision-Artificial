#!/usr/bin/env python
import cv2 as cv
import numpy as np
import time
import traceback
from collections import deque

# Matriz intrínseca obtenida en el ejercicio de calibración.
K = np.array([
    [840.32,   0.00, 330.34],
    [  0.00, 832.56, 221.40],
    [  0.00,   0.00,   1.00]
], dtype=np.float64)

# Coeficientes de distorsión obtenidos de calib.txt (ejercicio de calibración)
distCoeffs = np.array([0.3784, -4.2817, -0.0042, 0.0119, 16.0687], dtype=np.float64)


MARCADOR = np.array([
    [0.0,   0.0, 0.0],  # Esquina superior-izquierda (origen)
    [0.0,  15.0, 0.0],  # Esquina inferior-izquierda del palo vertical
    [5.0,  15.0, 0.0],  # Esquina inferior-derecha del palo vertical
    [5.0,   5.0, 0.0],  # Esquina interior (la cóncava)
    [10.2,  5.0, 0.0],  # Esquina inferior-derecha del brazo horizontal
    [10.2,  0.0, 0.0],  # Esquina superior-derecha del brazo horizontal
], dtype=np.float64)

# Paleta de colores para los objetos virtuales (BGR)
COLORES = [
    (0, 255,   0),  
    (0, 128, 255),   
    (255, 0, 255),   
    (0, 255, 255),   
    (255, 0,   0),   
    (0, 0,   255),   
    (255, 255, 0),  
    (128, 255, 128),
    (255, 128, 255), 
]

# Suavizado: nº de poses a promediar para reducir jitter
SMOOTH_WINDOW = 5

# Frames máximos manteniendo la última pose si no se detecta el marcador
MAX_FRAMES_SIN_DETECCION = 45

# Factor de interpolación para animación (más alto = más rápido)
ALPHA_ANIMACION = 0.08

# Tamaño del cubo wireframe en unidades del marcador
TAM_CUBO = 4


def homog(x):
    """Convierte puntos ordinarios a coordenadas homogéneas."""
    ax = np.array(x)
    uc = np.ones(ax.shape[:-1] + (1,))
    return np.append(ax, uc, axis=-1)

def inhomog(x):
    """Convierte puntos homogéneos a coordenadas ordinarias."""
    ax = np.array(x)
    return ax[..., :-1] / ax[..., [-1]]

def htrans(h, x):
    """Aplica una transformación homogénea h a un conjunto de puntos."""
    return inhomog(homog(x) @ h.T)

def rgb2gray(x):
    return cv.cvtColor(x, cv.COLOR_BGR2GRAY)

def orientation(c):
    return cv.contourArea(c.astype(np.float32), oriented=True)

def redondez(c):
    p = cv.arcLength(c.astype(np.float32), closed=True)
    oa = orientation(c)
    return (oa, 100 * 4 * np.pi * abs(oa) / p**2) if p > 0 else (0, 0)

def boundingBox(c):
    (x1, y1), (x2, y2) = c.min(0), c.max(0)
    return (x1, y1), (x2, y2)

def internal(c, h, w):
    (x1, y1), (x2, y2) = boundingBox(c)
    return x1 > 1 and x2 < w - 2 and y1 > 1 and y2 < h - 2

def redu(c, eps=3.0):
    return cv.approxPolyDP(c, eps, True).reshape(-1, 2)

def polygons(cs, n, prec=3.0):
    rs = [redu(c, prec) for c in cs]
    return [r for r in rs if r.shape[0] == n]

def extractContours(g, minarea=10, minredon=25, reduprec=3.0):
    g_suavizada = cv.GaussianBlur(g, (5, 5), 0)

    _, gt = cv.threshold(g_suavizada, 189, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    contours = cv.findContours(gt, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]
    h, w = g.shape
    tharea = (min(h, w) * minarea / 100.)**2

    def good(c):
        oa, r = redondez(c)
        return oa > 0 and abs(oa) >= tharea and r > minredon

    buenos    = [c for c in contours if good(c)]
    aprox     = [redu(c.reshape(-1, 2), reduprec) for c in buenos]
    internos  = [c for c in aprox if internal(c, h, w)]

    return internos

def rmsreproj(view, model, transf):
    """Error RMS de reproyección."""
    err = view - htrans(transf, model)
    return np.sqrt(np.mean(err.flatten()**2))

def jc(*args):
    return np.hstack(args)

def pose(K, image, model):
    """
    Estima pose con solvePnP.
    Devuelve (rms, M, rvec, tvec). rvec/tvec se devuelven aplanados
    a 1D para facilitar el promediado posterior.
    """
    ok, rvec, tvec = cv.solvePnP(model, image, K, distCoeffs)
    if not ok:
        return 1e6, None, None, None
    R, _ = cv.Rodrigues(rvec)
    M = K @ jc(R, tvec)
    rms = rmsreproj(image, model, M)
    return rms, M, rvec.flatten(), tvec.flatten()

def rots(c):
    """Todas las rotaciones cíclicas de los vértices del polígono."""
    return [np.roll(c, k, 0) for k in range(len(c))]

def bestPose(K, view, model):
    """Prueba todas las asignaciones y devuelve la mejor."""
    poses = [pose(K, v.astype(float), model) for v in rots(view)]
    return sorted(poses, key=lambda p: p[0])[0]

def promediar_poses(poses_rt, K):
    """
    Promedia una lista de (rvec, tvec) por separado y reconstruye M.

    Frente a promediar K[R|t] directamente, esto garantiza que la
    rotación resultante sea siempre una rotación válida (ortogonal),
    porque la reconstruimos con cv.Rodrigues. Para variaciones pequeñas
    entre frames consecutivos (que es siempre el caso del suavizado),
    promediar rvec en el espacio axis-angle es equivalente, a primer
    orden, al promedio correcto sobre SO(3).
    """
    rvecs = np.array([rv for rv, _ in poses_rt])
    tvecs = np.array([tv for _, tv in poses_rt])
    rvec_avg = rvecs.mean(axis=0)
    tvec_avg = tvecs.mean(axis=0)
    R_avg, _ = cv.Rodrigues(rvec_avg)
    return K @ np.hstack([R_avg, tvec_avg.reshape(3, 1)])


class ObjetoVirtual:
    """
    Un cubo wireframe que vive en coordenadas 3D del marcador.
    Mantiene posición actual y destino, e interpola suavemente.
    """

    def __init__(self, posicion, color, tamano=TAM_CUBO):
        self.pos     = np.array(posicion, dtype=np.float64)
        self.destino = self.pos.copy()
        self.color   = color
        self.tamano  = tamano

    def set_destino(self, nuevo):
        """Asigna un nuevo destino (en coordenadas del marcador)."""
        self.destino = np.array(nuevo, dtype=np.float64)

    def update(self, alpha=ALPHA_ANIMACION):
        """Avanza la posición hacia el destino con interpolación suave."""
        self.pos += alpha * (self.destino - self.pos)

    def distancia_destino(self):
        return float(np.linalg.norm(self.destino - self.pos))

    def vertices_cubo(self):
        """
        Devuelve los 16 vértices del cubo wireframe (con repeticiones
        para que polylines lo dibuje en un solo trazo continuo).
        El cubo está centrado en self.pos en XY y apoyado sobre Z=0.
        """
        x, y, _ = self.pos
        s = self.tamano
        # 4 esquinas de la base inferior
        b00 = [x - s/2, y - s/2, 0]
        b10 = [x + s/2, y - s/2, 0]
        b11 = [x + s/2, y + s/2, 0]
        b01 = [x - s/2, y + s/2, 0]
        # 4 esquinas de la cara superior
        t00 = [x - s/2, y - s/2, s]
        t10 = [x + s/2, y - s/2, s]
        t11 = [x + s/2, y + s/2, s]
        t01 = [x - s/2, y + s/2, s]

        # Recorrido en un solo trazo (base -> tapa -> aristas verticales)
        return np.array([
            b00, b10, b11, b01, b00,    # base inferior
            t00, t10, t11, t01, t00,    # tapa
            t10, b10, b11, t11, t01, b01
        ], dtype=np.float64)

    def draw(self, frame, M):
        """Proyecta el cubo con M y lo dibuja sobre el frame."""
        verts = self.vertices_cubo()
        proy  = htrans(M, verts)
        cv.polylines(frame, [proy.astype(np.int32)], False, self.color, 2)

    def draw_destino(self, frame, M):
        """Dibuja un marcador visual en la posición de destino."""
        x, y, _ = self.destino
        s = self.tamano * 0.7
        cruz = np.array([
            [x - s/2, y, 0], [x + s/2, y, 0],
            [x, y - s/2, 0], [x, y + s/2, 0],
        ])
        p = htrans(M, cruz).astype(np.int32)
        cv.line(frame, tuple(p[0]), tuple(p[1]), self.color, 1)
        cv.line(frame, tuple(p[2]), tuple(p[3]), self.color, 1)

        # Círculo alrededor del destino, sólo si el objeto no ha llegado aún
        dist = self.distancia_destino()
        if dist > 0.02:
            centro = htrans(M, np.array([[x, y, 0]]))[0].astype(int)
            cv.circle(frame, tuple(centro), 8, self.color, 1, cv.LINE_AA)




class AplicacionRA:
    """
    Mantiene el estado del programa: objetos, pose actual, suavizado,
    y maneja los eventos de ratón y teclado.
    """

    def __init__(self):
        self.objetos              = []
        self.objeto_activo        = 0
        self.M                    = None    # pose suavizada actual
        self.H_pantalla_marcador  = None    # para des-proyectar clics
        # Historial de (rvec, tvec) para promediar correctamente la pose
        self.pose_history         = deque(maxlen=SMOOTH_WINDOW)
        self.frames_sin_deteccion = 0
        self.fps                  = 0.0
        self.prev_t               = time.time()

    def on_mouse(self, event, x, y, flags, param):
        if self.H_pantalla_marcador is None:
            return  # Sin marcador detectado, ignoramos los clics

        if event == cv.EVENT_LBUTTONDOWN:
            destino_3d = self._desproyectar(x, y)
            if not self.objetos:
                self._crear_objeto(destino_3d)
            else:
                self.objetos[self.objeto_activo].set_destino(destino_3d)

        elif event == cv.EVENT_RBUTTONDOWN:
            destino_3d = self._desproyectar(x, y)
            self._crear_objeto(destino_3d)

    def _desproyectar(self, x, y):
        """
        Convierte un píxel (x,y) a coordenadas 3D sobre el marcador.

        solvePnP interioriza la distorsión, así que la M resultante
        proyecta a píxeles *ideales* (sin distorsionar). Para que la
        homografía inversa funcione bien fuera del centro óptico,
        rectificamos primero el píxel del clic con undistortPoints.
        """
        pt     = np.array([[[float(x), float(y)]]], dtype=np.float32)
        pt_und = cv.undistortPoints(pt, K, distCoeffs, P=K)
        xu, yu = pt_und[0, 0]
        punto_2d = htrans(self.H_pantalla_marcador, (xu, yu))
        return np.array([punto_2d[0], punto_2d[1], 0.0])

    def _crear_objeto(self, pos):
        color = COLORES[len(self.objetos) % len(COLORES)]
        self.objetos.append(ObjetoVirtual(pos, color))
        self.objeto_activo = len(self.objetos) - 1
        print(f"Creado objeto #{self.objeto_activo} en "
              f"({pos[0]:.2f}, {pos[1]:.2f}) color {color}")

    def procesar(self, frame):
        now = time.time()
        dt  = now - self.prev_t
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.prev_t = now

        try:
            g = rgb2gray(frame)
            conts = extractContours(g, reduprec=3.0)
            hexs  = polygons(conts, 6)

            mejor_M    = None
            mejor_rvec = None
            mejor_tvec = None
            mejor_rms  = 1e6

            # Si hay varios hexágonos, nos quedamos con la mejor pose
            for poly in hexs:
                rms, M_pose, rvec_pose, tvec_pose = bestPose(K, poly, MARCADOR)
                if rms < mejor_rms:
                    mejor_rms  = rms
                    mejor_M    = M_pose
                    mejor_rvec = rvec_pose
                    mejor_tvec = tvec_pose

            # Si la pose es razonable, la incorporamos al historial
            if mejor_M is not None:
                if mejor_rms < 50.0:
                    self.pose_history.append((mejor_rvec, mejor_tvec))
                    self.frames_sin_deteccion = 0
                else:
                    self.frames_sin_deteccion += 1
            else:
                self.frames_sin_deteccion += 1

            # Promediamos las últimas poses para suavizar
            if (len(self.pose_history) > 0
                    and self.frames_sin_deteccion < MAX_FRAMES_SIN_DETECCION):
                self.M = promediar_poses(list(self.pose_history), K)

                # Homografía pantalla -> marcador (2D) para des-proyectar
                pts_marcador_2d  = MARCADOR[:, :2].astype(np.float32)
                pts_pantalla     = htrans(self.M, MARCADOR).astype(np.float32)
                self.H_pantalla_marcador, _ = cv.findHomography(
                    pts_pantalla, pts_marcador_2d
                )
            else:
                # marcador perdido durante demasiado tiempo
                self.M = None
                self.H_pantalla_marcador = None
                self.pose_history.clear()

        except Exception as e:
            print(f"Frame descartado: {e}")
            traceback.print_exc()
            self.M = None
            self.H_pantalla_marcador = None


        for obj in self.objetos:
            obj.update()

        if self.M is not None:
            contorno_proy = htrans(self.M, MARCADOR).astype(np.int32)
            cv.polylines(frame, [contorno_proy], True, (200, 200, 200),
                         1, cv.LINE_AA)

            for obj in self.objetos:
                obj.draw_destino(frame, self.M)
                obj.draw(frame, self.M)

        self._dibujar_hud(frame)

    def _dibujar_hud(self, frame):
        h, w = frame.shape[:2]
        cv.putText(frame, f"FPS: {self.fps:5.1f}", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)

        estado = "MARCADOR OK" if self.M is not None else "BUSCANDO MARCADOR..."
        col    = (0, 255, 0) if self.M is not None else (0, 0, 255)
        cv.putText(frame, estado, (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv.LINE_AA)

        cv.putText(frame, f"Objetos: {len(self.objetos)}", (10, 75),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

        if self.objetos:
            col_act = self.objetos[self.objeto_activo].color
            dist    = self.objetos[self.objeto_activo].distancia_destino()
            cv.putText(frame,
                       f"Activo: #{self.objeto_activo}  dist={dist:.2f}",
                       (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       col_act, 2, cv.LINE_AA)

        ayuda = "L-click: destino | R-click: nuevo | 1-9: activo | r: reset | q: salir"
        cv.putText(frame, ayuda, (10, h - 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
                   cv.LINE_AA)

    def manejar_tecla(self, key):
        if key == ord('r'):
            self.objetos = []
            self.objeto_activo = 0
            print("Reset: eliminados todos los objetos.")
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(self.objetos):
                self.objeto_activo = idx
                print(f"Objeto activo: #{idx}")



def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede abrir la webcam.")
        return

    app = AplicacionRA()
    cv.namedWindow("Desplazamiento de objetos")
    cv.setMouseCallback("Desplazamiento de objetos", app.on_mouse)

    print("Programa iniciado. Pulsa 'q' o ESC para salir.")
    print("Coloca el marcador en L delante de la cámara.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        app.procesar(frame)
        cv.imshow("Desplazamiento de objetos", frame)

        key = cv.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        app.manejar_tecla(key)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()