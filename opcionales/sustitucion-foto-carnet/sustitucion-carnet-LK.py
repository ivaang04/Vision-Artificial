#!/usr/bin/env python
import cv2
import numpy as np
import time
import sys

MIN_MATCH_COUNT        = 15
MIN_TRACKED_POINTS     = 12      
MAX_TRACKED_POINTS     = 200     # cap para evitar crecimiento descontrolado
REINJECT_INTERVAL      = 5       # buscar corners nuevos cada N frames
FULL_REDETECT_INTERVAL = 120     # ORB completo cada M frames (anti-drift)
LOWE_RATIO             = 0.75
RANSAC_THRESHOLD       = 5.0
ORB_FEATURES           = 1500
FB_THRESHOLD           = 1.0     # umbral forward-backward en píxeles

# Parámetros LK piramidal (siguiendo lk_track.py)
LK_PARAMS = dict(
    winSize  = (15, 15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Parámetros para Shi-Tomasi (goodFeaturesToTrack)
GFTT_PARAMS = dict(
    qualityLevel = 0.1,
    minDistance  = 10,
    blockSize    = 7
)


carnet_ref = cv2.imread('carnet-bus.jpeg', cv2.IMREAD_GRAYSCALE)
if carnet_ref is None:
    sys.exit("No se ha podido cargar 'carnet-bus.jpeg'.")

cara_nueva = cv2.imread('foto-ejemplo.jpg')
if cara_nueva is None:
    sys.exit("No se ha podido cargar 'foto-ejemplo.jpg'.")

h_cara, w_cara = cara_nueva.shape[:2]
h_ref,  w_ref  = carnet_ref.shape[:2]

# Coordenadas del hueco de la foto dentro del carnet (sup-izq, sup-der, inf-der, inf-izq)
pts_cara_ref = np.float32([
    [ 44, 143],
    [205, 143],
    [205, 344],
    [ 44, 344],
]).reshape(-1, 1, 2)

# Esquinas de la imagen de reemplazo
pts_cara_nueva = np.float32([
    [0,       0],
    [w_cara,  0],
    [w_cara,  h_cara],
    [0,       h_cara]
]).reshape(-1, 1, 2)


pts_carnet_ref = np.float32([
    [0,     0],
    [w_ref, 0],
    [w_ref, h_ref],
    [0,     h_ref]
]).reshape(-1, 1, 2)

# Detector ORB y matcher (solo en fase de detección/anclaje)
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
kp_ref, des_ref = orb.detectAndCompute(carnet_ref, None)
if des_ref is None:
    sys.exit("No se han detectado descriptores en la referencia.")
print(f"Keypoints en referencia: {len(kp_ref)}")
print(f"Carnet de referencia: {w_ref}x{h_ref}")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def detectar_orb(frame_gray):
    """
    Detección ORB + matching Lowe + homografía RANSAC.
    Devuelve los pares (ref_pts, tracked_pts) inliers, o (None, None) si falla.
    """
    kp_frame, des_frame = orb.detectAndCompute(frame_gray, None)
    if des_frame is None or len(kp_frame) < 2:
        return None, None

    matches = bf.knnMatch(des_ref, des_frame, k=2)
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    if len(good) <= MIN_MATCH_COUNT:
        return None, None

    src = np.float32([kp_ref[m.queryIdx].pt   for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESHOLD)
    if H is None or mask is None:
        return None, None

    inliers = mask.ravel().astype(bool)
    return src[inliers], dst[inliers]



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit(" No se ha podido abrir la webcam.")

# Estado persistente entre frames
ref_pts            = None   # puntos en la imagen de referencia
tracked_pts        = None   # esos mismos puntos en el frame actual
prev_gray          = None
H_last             = None   # última homografía válida (para reinyección)
frames_since_full  = 0

prev_t  = time.time()
fps     = 0.0
n_frame = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    n_frame += 1


    now = time.time()
    dt  = now - prev_t
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)
    prev_t = now

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    modo = "DETECT"

    try:

        need_full = (
            tracked_pts is None
            or len(tracked_pts) < MIN_TRACKED_POINTS
            or frames_since_full >= FULL_REDETECT_INTERVAL
        )

        if need_full:
            ref_pts, tracked_pts = detectar_orb(gray)
            frames_since_full = 0
            modo = "DETECT"
        else:


            p0 = tracked_pts
            p1, _, _  = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0,  None, **LK_PARAMS)
            p0r, _, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **LK_PARAMS)

            # Aceptamos solo los puntos cuyo recorrido ida-vuelta < 1 px
            d    = np.abs(p0 - p0r).reshape(-1, 2).max(axis=1)
            good = d < FB_THRESHOLD

            tracked_pts = p1[good].reshape(-1, 1, 2)
            ref_pts     = ref_pts[good].reshape(-1, 1, 2)
            frames_since_full += 1
            modo = "TRACK"


        # Homografía actual a partir de las correspondencias vivas

        if tracked_pts is not None and len(tracked_pts) >= 4:
            H, _ = cv2.findHomography(ref_pts, tracked_pts,
                                      cv2.RANSAC, RANSAC_THRESHOLD)

            if H is not None:
                pts_cara_frame = cv2.perspectiveTransform(pts_cara_ref, H)

                # Validación geométrica del polígono proyectado
                if (cv2.isContourConvex(np.int32(pts_cara_frame))
                        and cv2.contourArea(pts_cara_frame) > 500):

                    H_last = H  # se conserva para la reinyección

                    H_local = cv2.getPerspectiveTransform(pts_cara_nueva,
                                                          pts_cara_frame)
                    if H_local is not None:
                        h_f, w_f = frame.shape[:2]
                        cara_warped = cv2.warpPerspective(cara_nueva, H_local,
                                                          (w_f, h_f))

                        mask_render = np.zeros((h_f, w_f), dtype=np.uint8)
                        cv2.fillConvexPoly(mask_render,
                                           np.int32(pts_cara_frame), 255)
                        mask_inv = cv2.bitwise_not(mask_render)

                        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                        cara_fg  = cv2.bitwise_and(cara_warped, cara_warped,
                                                   mask=mask_render)
                        frame    = cv2.add(frame_bg, cara_fg)
                else:
                    # Polígono degenerado, forzar reset en el siguiente frame
                    tracked_pts = None

        # Reinyección incremental de corners
        if (modo == "TRACK"
                and H_last is not None
                and tracked_pts is not None
                and n_frame % REINJECT_INTERVAL == 0
                and len(tracked_pts) < MAX_TRACKED_POINTS):

            # Proyectamos el rectángulo del carnet completo al frame
            pts_carnet_frame = cv2.perspectiveTransform(pts_carnet_ref, H_last)

            # Máscara: blanco dentro del carnet, negro fuera y en torno a
            # los puntos vivos (radio 5, igual que en lk_track.py)
            mask_inject = np.zeros_like(gray)
            cv2.fillConvexPoly(mask_inject, np.int32(pts_carnet_frame), 255)
            for pt in tracked_pts.reshape(-1, 2):
                cv2.circle(mask_inject, tuple(np.int32(pt)), 5, 0, -1)

            # Cuantos podemos añadir sin pasar del cap
            cupo = MAX_TRACKED_POINTS - len(tracked_pts)
            new_corners = cv2.goodFeaturesToTrack(
                gray, mask=mask_inject, maxCorners=cupo, **GFTT_PARAMS
            )

            if new_corners is not None and len(new_corners) > 0:
                # Cada corner nuevo está en el frame actual; necesitamos su
                # coordenada equivalente en la imagen de referencia.
                # Eso se obtiene aplicando la homografía inversa.
                try:
                    H_inv = np.linalg.inv(H_last)
                    new_ref_pts = cv2.perspectiveTransform(new_corners, H_inv)

                    tracked_pts = np.vstack([tracked_pts, new_corners])
                    ref_pts     = np.vstack([ref_pts,     new_ref_pts])
                except np.linalg.LinAlgError:
                    pass  # H singular: saltamos la reinyección este frame

    except Exception as e:
        print(f"Frame descartado: {e}")
        tracked_pts = None  

    prev_gray = gray


    cv2.putText(frame, f"FPS:  {fps:5.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    color_modo = (0, 255, 255) if modo == "TRACK" else (0, 165, 255)
    cv2.putText(frame, f"MODO: {modo}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_modo, 2, cv2.LINE_AA)
    n_pts = 0 if tracked_pts is None else len(tracked_pts)
    cv2.putText(frame, f"PTS:  {n_pts}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    cv2.imshow("Sustitucion foto carnet bus", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()