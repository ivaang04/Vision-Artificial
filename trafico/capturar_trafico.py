#!/usr/bin/env python

import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from umucv.stream import autoStream
from umucv.util import putText


# He creado una clase slider propia porque no me funciona bien la que hay en el repositorio
class CustomSlider:
    def __init__(self, name, vmin, vmax, default, step=1):
        self.name     = name
        self.vmin     = vmin
        self.vmax     = vmax
        self.value    = default
        self.step     = step
        self.dragging = False
        self.x = self.y = self.w = self.h = 0

    def set_rect(self, x, y, w=200, h=18):
        self.x, self.y, self.w, self.h = x, y, w, h

    def _val_to_px(self, val):
        ratio = (val - self.vmin) / (self.vmax - self.vmin)
        return int(self.x + ratio * self.w)

    def _px_to_val(self, px):
        ratio   = np.clip((px - self.x) / self.w, 0, 1)
        raw     = self.vmin + ratio * (self.vmax - self.vmin)
        snapped = round(raw / self.step) * self.step
        return int(np.clip(snapped, self.vmin, self.vmax))

    def handle_mouse(self, event, mx, my):
        hx  = self._val_to_px(self.value)
        mid = self.y + self.h // 2
        hit = abs(mx - hx) < 10 and abs(my - mid) < 12
        if event == cv.EVENT_LBUTTONDOWN and hit:
            self.dragging = True
        elif event == cv.EVENT_MOUSEMOVE and self.dragging:
            self.value = self._px_to_val(mx)
        elif event == cv.EVENT_LBUTTONUP:
            self.dragging = False

    def draw(self, img, color=(200, 200, 200)):
        mid = self.y + self.h // 2
        hx  = self._val_to_px(self.value)
        cv.line(img, (self.x, mid), (self.x + self.w, mid), (70, 70, 70), 2, cv.LINE_AA)
        cv.line(img, (self.x, mid), (hx, mid), color, 3, cv.LINE_AA)
        cv.circle(img, (hx, mid), 8, color, -1, cv.LINE_AA)
        cv.circle(img, (hx, mid), 8, (25, 25, 25), 1, cv.LINE_AA)
        cv.putText(img, f"{self.name}: {self.value}",
                   (self.x, self.y - 6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv.LINE_AA)


NOMBRE_VENTANA = 'Analisis de trafico'
cv.namedWindow(NOMBRE_VENTANA)

sliders = [
    CustomSlider('Area Minima',  50, 3000, 500, step=50),
    CustomSlider('Cierre Morf',   1,   15,   5, step=1),
    CustomSlider('Max Dist',     20,  200,  80, step=10),
]
COLORES_SLIDER = [(80, 220, 80), (255, 220, 50), (60, 180, 255)]


roi_start  = None
roi_active = [] # Guardará [x1, y1, x2, y2]

# Al los sliders estar en la propia ventana, puede colisionar con los clics que hagamos para colocar el ROI, por tanto,
#tendremos que llevar cuidado.
def mouse_cb(event, mx, my, flags, param):
    global roi_start, roi_active
    
    # Comprobar si estamos tocando un slider
    tocando_slider = False
    for s in sliders:
        s.handle_mouse(event, mx, my)
        if s.dragging:
            tocando_slider = True

    # Si no tocamos sliders, gestionamos el dibujo del ROI
    if not tocando_slider:
        if event == cv.EVENT_LBUTTONDOWN:
            roi_start = (mx, my)
            roi_active = []
        elif event == cv.EVENT_MOUSEMOVE and roi_start is not None:
            roi_active = [roi_start[0], roi_start[1], mx, my]
        elif event == cv.EVENT_LBUTTONUP:
            if roi_start is not None:
                x1, y1 = roi_start
                x2, y2 = mx, my
                # Solo guardamos el ROI si tiene un tamaño mínimo
                if abs(x2 - x1) > 20 and abs(y2 - y1) > 20:
                    roi_active = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                else:
                    roi_active = []
                roi_start = None
        elif event == cv.EVENT_RBUTTONDOWN:
            # Clic derecho para borrar el ROI
            roi_active = []
            roi_start = None

cv.setMouseCallback(NOMBRE_VENTANA, mouse_cb)

bgsub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=12, detectShadows=True)


#Variables globales

count_izquierda = 0
count_derecha   = 0
next_id         = 0
tracked_objects = {}
DIR_HISTORY_LEN = 5

frames_procesados   = 0
VENTANA_TIEMPO      = 100
historial_izquierda = []
historial_derecha   = []
flujo_temporal_izq  = 0
flujo_temporal_der  = 0

#Bucle principal

for key, frame in autoStream():
    frames_procesados += 1
    alto, ancho = frame.shape[:2]
    frame_display = frame.copy()

    # Defino una linea central que servirá para determinar si un vehículo está yendo por el carril izquierdo o derecho,
    # es decir, una vez que cruce la línea central se verá su carril, y dependiendo del que sea, aumentará un contador u otro.
    LINEA_X = ancho // 2

    # Panel de sliders
    panel_x = ancho - 250
    for i, s in enumerate(sliders):
        s.set_rect(panel_x + 40, alto - 35 - i * 50, w=185)

    MIN_AREA     = sliders[0].value
    MORF_SIZE    = max(1, sliders[1].value)
    MAX_DISTANCE = sliders[2].value

    # Procesamiento de la imagen y ROI
    fgmask = bgsub.apply(frame)
    _, fgmask = cv.threshold(fgmask, 200, 255, cv.THRESH_BINARY)
    
    # Aplicar el ROI dibujado por el usuario
    if len(roi_active) == 4:
        x1, y1, x2, y2 = roi_active
        mask_roi = np.zeros((alto, ancho), dtype=np.uint8)
        cv.rectangle(mask_roi, (x1, y1), (x2, y2), 255, -1)
        # Tapamos lo que quede fuera (ej. el mar)
        fgmask = cv.bitwise_and(fgmask, mask_roi)
        
        cv.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 255), 2)
        putText(frame_display, "ZONA DE ANALISIS", (x1, y1 - 10), color=(0, 255, 255))
    else:
        putText(frame_display, "DIBUJA UN AREA CON EL RATON", (20, alto - 20), color=(0, 0, 255))

    # Al definir una matriz marcadamente asimétrica de # 7 de alto por 2 de ancho. se fuerza al algoritmo a buscar y preservar únicamente estructuras con dominancia vertical en la imagen.
     # Es un corte muy fino y muy largo de arriba a abajo.
    # Destruye instantáneamente cualquier conexión horizontal entre carriles.
    kernel_open = np.ones((7, 2), np.uint8) 
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel_open)
    

    # Una vez separados los vehículos paralelos, aplicamos un kernel cuadrado 
    # cuyo tamaño (MORF_SIZE) controla el usuario. Esto consolida los fragmentos 
    # internos de una misma carrocería (ej. separación por el parabrisas) en una 
    # masa compacta y sólida, sin desbordarse hacia el carril vecino.
    kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (MORF_SIZE, MORF_SIZE))
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel_close)

    # Componentes Conexas
    n, cc, st, cen = cv.connectedComponentsWithStats(fgmask)
    current_centroids = []

    for i in range(1, n):
        area = st[i][cv.CC_STAT_AREA]
        if area < MIN_AREA: continue

        x1, y1, w, h = st[i][:4]
        cx, cy = int(cen[i][0]), int(cen[i][1])
        
        current_centroids.append((cx, cy))
        cv.rectangle(frame_display, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

# Tracking
    new_tracked = {}
    used_ids    = set()
    MARGEN = 15
    MAX_FRAMES_PERDIDO = 15  # Frames que el programa recuerda a un coche invisible

    # Actualizamos los coches que si vemos en este fotograma
    for cx, cy in current_centroids:
        best_id, best_dist = None, float('inf')
        for tid, obj in tracked_objects.items():
            if tid in used_ids: continue
            dist = math.hypot(cx - obj['px'], cy - obj['py'])
            if dist < MAX_DISTANCE and dist < best_dist:
                best_dist, best_id = dist, tid

        if best_id is not None:
            # Reenganchamos un coche existente
            obj = tracked_objects[best_id]
            ya_contado = obj.get('counted', False)
            x_antiguo = obj['px']
            
            if not ya_contado:
                # Cruce reforzado con el margen de cortesía
                cruza_der = x_antiguo < (LINEA_X + MARGEN) and cx >= LINEA_X
                cruza_izq = x_antiguo > (LINEA_X - MARGEN) and cx <= LINEA_X
                
                if cruza_der and x_antiguo < LINEA_X:
                    count_derecha += 1; flujo_temporal_der += 1
                    ya_contado = True
                    cv.circle(frame_display, (cx, cy), 40, (0, 165, 255), -1) # Naranja
                    
                elif cruza_izq and x_antiguo > LINEA_X:
                    count_izquierda += 1; flujo_temporal_izq += 1
                    ya_contado = True
                    cv.circle(frame_display, (cx, cy), 40, (0, 255, 0), -1) # Verde

            # Lo guardamos y reseteamos su contador de pérdida a 0
            new_tracked[best_id] = {'px': cx, 'py': cy, 'counted': ya_contado, 'perdido': 0}
            used_ids.add(best_id)
        else:
            # Es un coche totalmente nuevo
            new_tracked[next_id] = {'px': cx, 'py': cy, 'counted': False, 'perdido': 0}
            next_id += 1

    # "Salvar" a los coches que han "parpadeado"
    for tid, obj in tracked_objects.items():
        if tid not in used_ids:
            perdido = obj.get('perdido', 0) + 1
            if perdido < MAX_FRAMES_PERDIDO:
                #Lo mantenemos en cuenta en la lista con su última posición conocida
                obj['perdido'] = perdido
                new_tracked[tid] = obj

    tracked_objects = new_tracked

    # Gráficas
    if frames_procesados % VENTANA_TIEMPO == 0:
        historial_izquierda.append(flujo_temporal_izq)
        historial_derecha.append(flujo_temporal_der)
        flujo_temporal_izq, flujo_temporal_der = 0, 0

    if frames_procesados % VENTANA_TIEMPO == 0 and len(historial_izquierda) > 1:
        fig = plt.figure(figsize=(5, 3), dpi=80)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        t = np.arange(len(historial_izquierda))
        ax.plot(t, historial_izquierda, 'o-', color='orange', label='Izquierda')
        ax.plot(t, historial_derecha,   'x-', color='blue',   label='Derecha')
        ax.set_title('Flujo en tiempo real')
        ax.legend(); ax.grid(True)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        img_plot = cv.cvtColor(rgba, cv.COLOR_RGBA2BGR)
        cv.imshow('Grafica en tiempo de ejecucion', img_plot)
        plt.close(fig)

    #  Interfaz
    cv.line(frame_display, (LINEA_X, 0), (LINEA_X, alto), (0, 255, 255), 2)
    putText(frame_display, f"IZQ: {count_izquierda}  DER: {count_derecha}", (20, 40), color=(0, 255, 255))

    panel_top = alto - 35 - (len(sliders) - 1) * 50 - 25
    overlay   = frame_display.copy()
    cv.rectangle(overlay, (panel_x, panel_top), (ancho - 5, alto - 5), (15, 15, 15), -1)
    cv.addWeighted(overlay, 0.55, frame_display, 0.45, 0, frame_display)
    cv.rectangle(frame_display, (panel_x, panel_top), (ancho - 5, alto - 5), (80, 80, 80), 1)

    for s, col in zip(sliders, COLORES_SLIDER):
        s.draw(frame_display, color=col)

    cv.imshow(NOMBRE_VENTANA, frame_display)
    if key in (27, ord('q')): break

cv.destroyAllWindows()

# Generación de gráficas finales
print("\nGenerando gráficas finales...")
if len(historial_izquierda) > 0:
    t = np.arange(len(historial_izquierda))
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(t, historial_izquierda, label='Hacia izquierda', marker='o', color='orange')
    plt.plot(t, historial_derecha, label='Hacia derecha', marker='x', color='blue')
    plt.title('Vehículos por unidad de tiempo')
    plt.xlabel(f'Bloques de {VENTANA_TIEMPO} frames')
    plt.ylabel('Coches detectados')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    total = np.array(historial_izquierda) + np.array(historial_derecha)
    plt.bar(t, total, color='skyblue', label='Flujo total')
    plt.axhline(total.mean(), color='red', linestyle='--', label='Media (Hora punta)')
    plt.title('Intensidad de tráfico')
    plt.legend(); plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig('grafica_trafico.png')
    print(" -> Gráfica guardada como 'grafica_trafico.png'")
    plt.show()
