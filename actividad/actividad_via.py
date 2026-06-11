#!/usr/bin/env python

import cv2 as cv
import numpy as np
import time
import traceback
from threading import Thread
import datetime
from collections import deque
from ultralytics import YOLO
from umucv.stream import autoStream
from umucv.util import ROI, putText

import requests
from dotenv import load_dotenv
import os

# Cargamos las credenciales desde el archivo token.env
load_dotenv('token.env')
TELEGRAM_TOKEN = os.environ.get('TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('USER_ID')

FPS = 30
SEGUNDOS_PRE_EVENTO = 3
SEGUNDOS_POST_EVENTO = 2
buffer_frames = deque(maxlen=FPS * SEGUNDOS_PRE_EVENTO)

model = YOLO('yolov8n.pt') 
CLASE_PERSONA = 0 

grabando = False
frames_grabados = 0
zona_vigilancia = None  
bgsub = None            
MIN_AREA_MOVIMIENTO = 500


goon = True
alarma_pendiente = False
frame_para_enviar = None

def enviar_alerta_telegram(frame, mensaje):
    """
    Envía la foto directamente a la API de Telegram.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(" Error: Faltan las credenciales en token.env")
        return

    # Convertimos el frame a JPG
    exito, img_codificada = cv.imencode('.jpg', frame)
    
    if exito:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        archivos = {'photo': ('alerta.jpg', img_codificada.tobytes(), 'image/jpeg')}
        datos = {'chat_id': TELEGRAM_CHAT_ID, 'caption': mensaje}
        
        try:
            requests.post(url, files=archivos, data=datos)
            print(" -> Notificación de Telegram enviada con éxito.")
        except Exception as e:
            print(f" -> Error de conexión con Telegram: {e}")

def in_thread():
    """ Hilo secundario que está siempre vivo esperando alarmas """
    global alarma_pendiente, frame_para_enviar, goon
    
    while goon:
        if alarma_pendiente and frame_para_enviar is not None:
            print(' Hilo: Iniciando envío de Telegram...')
            enviar_alerta_telegram(frame_para_enviar, " ALERTA: Persona detectada en la zona.")
            print(' Hilo: Envío finalizado.')
            alarma_pendiente = False
            frame_para_enviar = None
        
        time.sleep(0.1) 

t = Thread(target=in_thread, daemon=True)
t.start()


NOMBRE_VENTANA = "Vigilancia"
cv.namedWindow(NOMBRE_VENTANA)
region_interactiva = ROI(NOMBRE_VENTANA)

print("\n === SISTEMA INICIADO ===")
print(" -> Dibuja con el ratón la zona de interés.")
print(" -> Pulsa 'c' para confirmar la zona y activar la alarma.")
print(" -> Pulsa 'x' para borrar la zona.")
print(" -> Pulsa 'q' o ESC para salir.")


# Bucle principal

try:
    for key, frame in autoStream():
        alto, ancho = frame.shape[:2]
        frame_procesado = frame.copy()
        evento_confirmado = False
        
        # Interfaz y ROI
        if region_interactiva.roi:
            x1, y1, x2, y2 = region_interactiva.roi
            cv.rectangle(frame_procesado, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            putText(frame_procesado, "PULSA 'c' PARA ACTIVAR ALARMA", (x1, y1 - 10))
            
            if key == ord('c'):
                rx, ry = min(x1, x2), min(y1, y2)
                rw, rh = abs(x2 - x1), abs(y2 - y1)
                
                if rw > 0 and rh > 0:
                    zona_vigilancia = [rx, ry, rw, rh]
                    region_interactiva.roi = [] 
                    bgsub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
                    print(f" Zona construida en: {zona_vigilancia}")

        if key == ord('x'):
            zona_vigilancia = None
            bgsub = None
            grabando = False
            print(" Zona de vigilancia desactivada.")

        # Lógica de detección y vigilancia
        if zona_vigilancia is not None and bgsub is not None:
            rx, ry, rw, rh = zona_vigilancia
            
            roi_img = frame[ry:ry+rh, rx:rx+rw]
            fgmask = bgsub.apply(roi_img)
            _, fgmask = cv.threshold(fgmask, 200, 255, cv.THRESH_BINARY)
            fgmask = cv.dilate(fgmask, np.ones((5,5), np.uint8), iterations=1)
            
            contornos, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            hay_movimiento = any(cv.contourArea(c) > MIN_AREA_MOVIMIENTO for c in contornos)
            
            if hay_movimiento or grabando:
                resultados = model(frame_procesado, stream=True, verbose=False)
                
                for r in resultados:
                    for caja in r.boxes:
                        clase_id = int(caja.cls[0])
                        
                        if clase_id == CLASE_PERSONA:
                            x1 = max(0, int(caja.xyxy[0][0]))
                            y1 = max(0, int(caja.xyxy[0][1]))
                            x2 = min(ancho, int(caja.xyxy[0][2]))
                            y2 = min(alto, int(caja.xyxy[0][3]))
                            
                            persona = frame_procesado[y1:y2, x1:x2]
                            if persona.size > 0:
                                frame_procesado[y1:y2, x1:x2] = cv.GaussianBlur(persona, (51, 51), 0)
                            
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            if (rx < cx < rx + rw) and (ry < cy < ry + rh):
                                evento_confirmado = True
                                cv.rectangle(frame_procesado, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                putText(frame_procesado, "PERSONA DETECTADA", (x1, y1 - 10))

            overlay = frame_procesado.copy()
            cv.rectangle(overlay, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), -1)
            cv.addWeighted(overlay, 0.2, frame_procesado, 0.8, 0, frame_procesado)
            cv.rectangle(frame_procesado, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)

        else:
            if not region_interactiva.roi:
                putText(frame_procesado, "SISTEMA DESARMADO: Dibuja zona", (10, 30))

        buffer_frames.append(frame_procesado.copy())

        # Grabación y delegación al hilo secundario
        if evento_confirmado and not grabando:
            print("\n Evento detectado. Grabando vídeo...")
            grabando = True
            frames_grabados = 0
            
            if not alarma_pendiente:
                frame_para_enviar = frame_procesado.copy()
                alarma_pendiente = True

            marca_tiempo = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            nombre_video = f"evento_{marca_tiempo}.mp4"
            
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            out = cv.VideoWriter(nombre_video, fourcc, FPS, (ancho, alto))
            
            for f in buffer_frames:
                out.write(f)
                
        if grabando:
            out.write(frame_procesado)
            frames_grabados += 1
            if evento_confirmado:
                frames_grabados = 0

            cv.circle(frame_procesado, (30, alto - 30), 10, (0, 0, 255), -1)
            putText(frame_procesado, "REC", (50, alto - 35))

            if frames_grabados >= FPS * SEGUNDOS_POST_EVENTO:
                out.release()
                print(" Vídeo guardado. Alarma reactivada.")
                grabando = False 
                
        cv.imshow(NOMBRE_VENTANA, frame_procesado)

        if key in (27, ord('q')):
            break

except KeyboardInterrupt:
    print("\n Detenido por el usuario (Ctrl+C).")
    goon = False
    raise

except Exception as e:
    print("\nError: Ha ocurrido una excepción:", e)
    traceback.print_exc()
    goon = False      

finally:
    goon = False 
    if grabando:
        out.release()
    cv.destroyAllWindows()
    print("Programa finalizado limpiamente.")