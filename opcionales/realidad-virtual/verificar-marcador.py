#!/usr/bin/env python
import cv2 as cv
import numpy as np

def rgb2gray(x):
    return cv.cvtColor(x, cv.COLOR_RGB2GRAY)

def orientation(x):
    return cv.contourArea(x.astype(np.float32), oriented=True)

def redondez(c):
    p = cv.arcLength(c.astype(np.float32), closed=True)
    oa = orientation(c)
    if p > 0:
        return oa, 100 * 4 * np.pi * abs(oa) / p**2
    return 0, 0

def boundingBox(c):
    (x1, y1), (x2, y2) = c.min(0), c.max(0)
    return (x1, y1), (x2, y2)

def internal(c, h, w):
    (x1, y1), (x2, y2) = boundingBox(c)
    return x1 > 1 and x2 < w - 2 and y1 > 1 and y2 < h - 2

def redu(c, eps=0.5):
    red = cv.approxPolyDP(c, eps, True)
    return red.reshape(-1, 2)

def polygons(cs, n, prec=2):
    rs = [redu(c, prec) for c in cs]
    return [r for r in rs if r.shape[0] == n]

def extractContours(g, minarea=10, minredon=25, reduprec=1):
    ret, gt = cv.threshold(g, 189, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours = cv.findContours(gt, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]
    h, w = g.shape
    tharea = (min(h, w) * minarea / 100.)**2

    def good(c):
        oa, r = redondez(c)
        black = oa > 0
        return black and abs(oa) >= tharea and r > minredon

    ok = [redu(c.reshape(-1, 2), reduprec) for c in contours if good(c)]
    return [c for c in ok if internal(c, h, w)]


cap = cv.VideoCapture(0)
print("Pulsa 'q' o ESC para salir.")
print("Coloca el folio con la 'L' frente a la cámara.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    g = rgb2gray(frame)
    g = cv.GaussianBlur(g, (5, 5), 0)
    conts = extractContours(g, reduprec=3)
    hexagonos = polygons(conts, 6)  


    for c in conts:
        cv.polylines(frame, [c.astype(np.int32)], True, (255, 0, 0), 1)


    for poly in hexagonos:
        cv.polylines(frame, [poly.astype(np.int32)], True, (0, 255, 0), 3)
        for i, (x, y) in enumerate(poly.astype(np.int32)):
            cv.circle(frame, (x, y), 6, (0, 0, 255), -1)
            cv.putText(frame, str(i), (x + 8, y - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


    cv.putText(frame, f"Contornos: {len(conts)}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(frame, f"Hexagonos (n=6): {len(hexagonos)}", (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv.imshow("Verificacion marcador L", frame)
    if cv.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv.destroyAllWindows()
