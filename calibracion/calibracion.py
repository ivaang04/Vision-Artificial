#!/usr/bin/env python

import math
import cv2 as cv
import numpy as np
from collections import deque         
from umucv.stream import autoStream
from umucv.util import putText

# Creo una clase Slider porque no me funciona bien con los sliders que hay en el repositorio.
class Slider:
    def __init__(self, label, vmin, vmax, default, step=1):
        self.label = label
        self.vmin, self.vmax = vmin, vmax
        self.value = default
        self.step = step
        self.dragging = False
        self.bx = self.by = self.bw = self.bh = 0

    def draw(self, img, row_y, w_panel):
        BAR_X, BAR_W, BAR_H = 150, w_panel - 150 - 20, 6
        self.bx, self.by, self.bw, self.bh = BAR_X, row_y - BAR_H//2, BAR_W, BAR_H
        
        cv.putText(img, self.label, (10, row_y + 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv.LINE_AA)
        txt_val = f"({self.value:03d}/{self.vmax:03d})"
        cv.putText(img, txt_val, (70, row_y + 5), cv.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv.LINE_AA)
        
        for i in range(0, self.bw, 10):
            cv.line(img, (self.bx + i, self.by + 10), (self.bx + i, self.by + 12), (150, 100, 50), 1)

        cv.line(img, (self.bx, self.by + 4), (self.bx + self.bw, self.by + 4), (180, 180, 180), 1)
        fill_x = int(self.bx + (self.value - self.vmin) / (self.vmax - self.vmin) * self.bw)
        cv.rectangle(img, (self.bx, self.by + 2), (fill_x, self.by + 6), (200, 150, 100), -1)
        cv.rectangle(img, (fill_x - 4, self.by), (fill_x + 4, self.by + 8), (255, 255, 255), -1)
        cv.rectangle(img, (fill_x - 4, self.by), (fill_x + 4, self.by + 8), (150, 150, 150), 1)

    def handle_mouse(self, event, mx, my):
        if event == cv.EVENT_LBUTTONDOWN:
            if self.bx <= mx <= self.bx + self.bw and self.by - 10 <= my <= self.by + self.bh + 10:
                self.dragging = True
        elif event == cv.EVENT_MOUSEMOVE and self.dragging:
            ratio = np.clip((mx - self.bx) / self.bw, 0, 1)
            self.value = int(round((self.vmin + ratio * (self.vmax - self.vmin)) / self.step) * self.step)
        elif event == cv.EVENT_LBUTTONUP: self.dragging = False


# Se carga el archivo donde está la matriz K y D

calibdatos = np.loadtxt("calib.txt")
K, D = calibdatos[:9].reshape(3, 3), calibdatos[9:]
W, H_img = 640, 480
cx, cy = K[0, 2], K[1, 2]

puntos_medida = deque(maxlen=2)
mouse_pos = (0, 0)

sliders = [
    Slider('fov', 0, 80, 56), 
    Slider('Z', 0, 290, 32),
    Slider('A', 0, 20, 8),
    Slider('X', 0, 100, 0)
]

# Hago un manejador porque si no los sliders colisionan con la ventana.
def manejador(event, x, y, flags, param):
    global mouse_pos
    mouse_pos = (x, y)
    if y >= H_img:
        for s in sliders: s.handle_mouse(event, x, y - H_img)
    elif event == cv.EVENT_LBUTTONDOWN:
        puntos_medida.append((x, y))
    elif event == cv.EVENT_RBUTTONDOWN: puntos_medida.clear()

cv.namedWindow("medidor")
cv.setMouseCallback("medidor", manejador)

def pixel_to_3d(u, v, Z_m, focal):
    return np.array([(u - cx) * Z_m / focal, (v - cy) * Z_m / focal, Z_m])

def proyectar(X_m, Y_m, Z_m, focal):
    Z_m = max(Z_m, 0.1)
    return (int(focal * (X_m / Z_m) + cx), int(focal * (Y_m / Z_m) + cy))


# Bucle principal
for key, frame in autoStream():
    frame = cv.resize(cv.undistort(frame, K, D), (W, H_img))
    
    # Obtención de valores y conversión a metros
    fov_val, z_val, a_val, x_val = [s.value for s in sliders]
    Z_m, A_m, X_off_m = z_val/10.0, a_val/10.0, x_val/10.0

    focal = (W / 2) / math.tan(math.radians(fov_val / 2)) if fov_val > 0 else K[0,0]

    # Cuadrícula
    cv.line(frame, (0, int(cy)), (W, int(cy)), (255, 255, 255), 1) # Horizonte
    
    # Pared y suelo
    for x in range(-5, 6):
        cv.line(frame, proyectar(x + X_off_m, A_m - 4, Z_m, focal), proyectar(x + X_off_m, A_m, Z_m, focal), (150, 150, 150), 1)
        cv.line(frame, proyectar(x + X_off_m, A_m, 0.5, focal), proyectar(x + X_off_m, A_m, Z_m, focal), (255, 255, 255), 1)

    # Líneas de altura y números
    for y_alt in range(0, 5):
        y_real = A_m - y_alt
        pt_izq = proyectar(-5 + X_off_m, y_real, Z_m, focal)
        pt_der = proyectar( 5 + X_off_m, y_real, Z_m, focal)
        cv.line(frame, pt_izq, pt_der, (255, 255, 255), 1)
        
        if 0 < pt_izq[1] < H_img:
            pos_n = proyectar(X_off_m, y_real, Z_m, focal)
            putText(frame, str(y_alt), (pos_n[0], pos_n[1] - 5))

    # Código para el "medidor"
    if len(puntos_medida) == 2:
        p1, p2 = puntos_medida
        dist = np.linalg.norm(pixel_to_3d(p1[0], p1[1], Z_m, focal) - 
                             pixel_to_3d(p2[0], p2[1], Z_m, focal)) * 100
        cv.line(frame, p1, p2, (0, 0, 255), 1)
        cv.circle(frame, p1, 3, (0,0,255), -1); cv.circle(frame, p2, 3, (0,0,255), -1)
        putText(frame, f"{dist:.0f} cm", tuple(np.mean([p1, p2], axis=0).astype(int)))

    putText(frame, f"FOV={fov_val:.1f} deg, f={int(focal)}px ({W}x{H_img})", (10, 25))
    putText(frame, f"Z={Z_m:.1f} m", (10, 50))
    putText(frame, f"alt={A_m:.1f} m", (10, 75))

    panel = np.ones((130, W, 3), dtype=np.uint8) * 240
    for i, s in enumerate(sliders): s.draw(panel, 20 + i*22, W)
    
    if mouse_pos[1] < H_img:
        b, g, r = frame[mouse_pos[1], mouse_pos[0]]
        txt_col = f"(x={mouse_pos[0]}, y={mouse_pos[1]}) ~ R:{r} G:{g} B:{b}"
        putText(panel, txt_col, (10, 115))

    cv.imshow("medidor", np.vstack([frame, panel]))
    if key in (27, ord('q')): break

cv.destroyAllWindows()