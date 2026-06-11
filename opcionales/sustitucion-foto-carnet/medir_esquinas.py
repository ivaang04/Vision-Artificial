#!/usr/bin/env python

import cv2

img = cv2.imread('carnet-bus.jpeg')
if img is None:
    raise FileNotFoundError("No se encuentra carnet-bus.jpeg")

print(f"Imagen: {img.shape[1]}x{img.shape[0]} px")
print("Haz clic en las 4 esquinas en el orden indicado:")
print("  1) superior-izquierda")
print("  2) superior-derecha")
print("  3) inferior-derecha")
print("  4) inferior-izquierda")
print("Pulsa 'q' para salir.\n")

puntos = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos.append((x, y))
        etiquetas = ["sup-izq", "sup-der", "inf-der", "inf-izq"]
        idx = len(puntos) - 1
        if idx < 4:
            print(f"  {etiquetas[idx]:8s} -> [{x}, {y}],")
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        if len(puntos) == 4:
            print("\nListo. Copia las 4 líneas dentro de pts_cara_ref en el script principal.")

cv2.namedWindow('Medir esquinas', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Medir esquinas', click)

while True:
    cv2.imshow('Medir esquinas', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
