#!/usr/bin/env python
from . import IReconocedor
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from umucv.util import check_and_download

class Reconocedor(IReconocedor): 
    def __init__(self):
        self.models = {}
        check_and_download(
            "embedder.tflite",
            "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite"
        )
        options = vision.ImageEmbedderOptions(
            base_options=python.BaseOptions(model_asset_path='embedder.tflite'),
            l2_normalize=True, 
            quantize=False
        )
        self.embedder = vision.ImageEmbedder.create_from_options(options)

        
    def precompute(self, image):
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mpimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        resultado = self.embedder.embed(mpimage)
        
        if len(resultado.embeddings) > 0:
            return resultado.embeddings[0].embedding
        return None
    
    
    def compare(self, features_actuales):
        if not self.models:
            return "Sin modelos", 1.0, [] # Devolvemos lista vacía para no romper el main

        resultados = []

        for nombre, features_modelo in self.models.items():
            similitud = features_actuales @ features_modelo
            distancia = 1.0 - similitud
            confianza = max(0, similitud) * 100 
            resultados.append((nombre, distancia, confianza))

        resultados.sort(key=lambda x: x[1])
        mejor_nombre, mejor_distancia, mejor_confianza = resultados[0]

        ranking_pantalla = [f"{n}: Dist {d:.2f} (Conf {c:.1f}%)" for n, d, c in resultados]

        if mejor_distancia > 0.4:
            return "Desconocido", mejor_distancia, ranking_pantalla

        return mejor_nombre, mejor_distancia, ranking_pantalla