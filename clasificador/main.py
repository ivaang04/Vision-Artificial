#!/usr/bin/env python

import cv2 as cv
import argparse
import os
import importlib
from umucv.stream import autoStream
from umucv.util import putText

# Configuración de argumentos
parser = argparse.ArgumentParser(description='Clasificador de imágenes modular')
parser.add_argument('--modelos', type=str, required=True, help='Carpeta con las imágenes de referencia')
parser.add_argument('--metodos', type=str, required=True, help='Nombre del script en la carpeta metodos (sin .py)')
args = parser.parse_args()

# Crear la carpeta de modelos si no existe
os.makedirs(args.modelos, exist_ok=True)

# Cargar el método dinámicamente
print(f"\nCargando método de comparación: {args.metodos}...")
try:
    modulo = importlib.import_module(f"metodos.{args.metodos}")
    clasificador = modulo.Reconocedor() 
except Exception as e:
    print(f"Error al cargar '{args.metodos}': {e}")
    exit()


print(f"Precomputando imágenes de: {args.modelos}...")
for archivo in os.listdir(args.modelos):
    if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
        ruta = os.path.join(args.modelos, archivo)
        img = cv.imread(ruta)
        
        if img is not None:
            nombre_modelo = os.path.splitext(archivo)[0]
            caracteristicas = clasificador.precompute(img) 
            
            if caracteristicas is not None:
                clasificador.models[nombre_modelo] = caracteristicas
                print(f"  -> '{nombre_modelo}' listo.")

print(" -> Pulsa 's' para guardar lo que ve la cámara como un nuevo modelo.")
print(" -> Pulsa ESC para salir.\n")

contador_nuevos = 1

# Bucle principal
for key, frame in autoStream():
    frame_limpio = frame.copy()
    caracteristicas_actuales = clasificador.precompute(frame)

    if caracteristicas_actuales is not None:
        mejor_modelo, distancia, ranking = clasificador.compare(caracteristicas_actuales)

        if mejor_modelo:
            texto = f"Ganador: {mejor_modelo} (Dist: {distancia:.2f})"
    
            modelo_seguro = mejor_modelo.lower()
            
            if "desconocido" in modelo_seguro or "sin modelos" in modelo_seguro:
                color = (0, 165, 255) 
            else:
                color = (0, 255, 0)

            putText(frame, texto, (20, 40), color=color)
            
            y_offset = 80 
            for linea_texto in ranking:
                putText(frame, linea_texto, (20, y_offset), color=(255, 255, 255))
                y_offset += 25 
        
        if key == ord('s'):
            nombre_nuevo = f"captura_manual_{contador_nuevos}"
            clasificador.models[nombre_nuevo] = caracteristicas_actuales
            ruta_guardado = os.path.join(args.modelos, f"{nombre_nuevo}.jpg")
            
            cv.imwrite(ruta_guardado, frame_limpio) 
            
            print(f" Nuevo modelo guardado: {nombre_nuevo}")
            contador_nuevos += 1
            
    else:
        putText(frame, "No detecto caracteristicas", (20, 40), color=(0, 0, 255))

    cv.imshow('Clasificador', frame)
    
    if key == 27: # ESC
        break

cv.destroyAllWindows()
