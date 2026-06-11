#!/usr/bin/env python
import cv2
import numpy as np
import time
import sys


MIN_MATCH_COUNT   = 17    # Mínimo de buenos matches para aceptar homografía
LOWE_RATIO        = 0.75    # Umbral del Lowe's Ratio Test
RANSAC_THRESHOLD  = 5.0     # Umbral de reproyección RANSAC (en píxeles)
ORB_FEATURES      = 2000    # Nº máximo de features para ORB



# Cargar imagen de referencia del carnet en escala de grises
carnet_ref = cv2.imread('carnet-bus.jpeg', cv2.IMREAD_GRAYSCALE)
if carnet_ref is None:
    sys.exit("No se ha podido cargar la imagen del carnet.")

# Cargar imagen de reemplazo (en color)
cara_nueva = cv2.imread('foto-ejemplo.jpg')
if cara_nueva is None:
    sys.exit("No se ha podido cargar la foto de ejemplo.")

h_cara, w_cara = cara_nueva.shape[:2]
h_ref,  w_ref  = carnet_ref.shape[:2]
print(f"Carnet de referencia: {w_ref}x{h_ref} px")
print(f"Cara nueva:           {w_cara}x{h_cara} px")


pts_cara_ref = np.float32([
    [44, 143],   # esquina superior-izquierda  
    [205, 143],   # esquina superior-derecha    
    [205, 344],  # esquina inferior-derecha    
    [44, 344],   # esquina inferior-izquierda
]).reshape(-1, 1, 2)

# Esquinas de la imagen de reemplazo (mismo orden que pts_cara_ref)
pts_cara_nueva = np.float32([
    [0,        0],
    [w_cara,   0],
    [w_cara,   h_cara],
    [0,        h_cara]
]).reshape(-1, 1, 2)


orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

# Keypoints y descriptores de la referencia (solo se calculan una vez)
kp_ref, des_ref = orb.detectAndCompute(carnet_ref, None)
if des_ref is None or len(kp_ref) == 0:
    sys.exit("No se han detectado descriptores en el carnet de referencia.")
print(f"Keypoints en referencia: {len(kp_ref)}")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("No se ha podido abrir la webcam.")

prev_t = time.time()
fps    = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se ha podido leer un frame.")
        break


    now = time.time()
    dt  = now - prev_t
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)
    prev_t = now


    try:
        # Convertir a escala de grises para el detector
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Keypoints y descriptores en el frame actual
        kp_frame, des_frame = orb.detectAndCompute(frame_gray, None)

        # Comprobaciones de seguridad: ORB puede no encontrar nada
        # en frames borrosos, muy oscuros, etc.
        if des_frame is None or len(kp_frame) < 2:
            raise ValueError("Descriptores insuficientes en el frame")

        matches = bf.knnMatch(des_ref, des_frame, k=2)

        # Lowe's Ratio Test, nos quedamos solo con matches muy discriminantes
        good_matches = []
        for pair in matches:
            if len(pair) < 2:           # algunos pares pueden tener un solo vecino
                continue
            m, n = pair
            if m.distance < LOWE_RATIO * n.distance:
                good_matches.append(m)


        if len(good_matches) > MIN_MATCH_COUNT:
            # Coordenadas de los matches en ambas imágenes
            src_pts = np.float32(
                [kp_ref[m.queryIdx].pt   for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp_frame[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)


            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)

            if H is not None:
                # Proyectamos las 4 esquinas de la foto al frame
                pts_cara_frame = cv2.perspectiveTransform(pts_cara_ref, H)


                # Como son 4 puntos exactos, getPerspectiveTransform es lo idóneo.
                H_local = cv2.getPerspectiveTransform(pts_cara_nueva,
                                                      pts_cara_frame)

                if H_local is not None:
                    h_f, w_f = frame.shape[:2]

                    # Warpeamos la cara_nueva al tamaño/perspectiva del frame
                    cara_warped = cv2.warpPerspective(cara_nueva, H_local,
                                                      (w_f, h_f))
                    
                    # Máscara negra del tamaño del frame
                    mask = np.zeros((h_f, w_f), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, np.int32(pts_cara_frame), 255)

                    mask_inv = cv2.bitwise_not(mask)

                    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

                    cara_fg  = cv2.bitwise_and(cara_warped, cara_warped,
                                               mask=mask)

                    # Composición final
                    frame = cv2.add(frame_bg, cara_fg)


    except Exception as e:
        # Lo notificamos solo en consola.
        print(f"Frame descartado: {e}")

    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Sustitucion foto carnet bus", frame)

    # Salir con 'q' o ESC
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()