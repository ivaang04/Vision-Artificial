#!/usr/bin/env python
from . import IReconocedor
import cv2 as cv
import numpy as np
import mediapipe as mp
from scipy.spatial import procrustes

class Reconocedor(IReconocedor):
    def __init__(self):
        self.models = {}
        self.mp_hands = mp.solutions.hands
        
        # Le decimos que busque hasta 2 manos como medida de robustez
        self.hands = self.mp_hands.Hands(
            static_image_mode=True, 
            max_num_hands=2, 
            min_detection_confidence=0.5
        )

    def precompute(self, image):
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        resultados = self.hands.process(rgb_image)
        
        if resultados.multi_hand_landmarks:

            # Si ve más de una mano en la imagen, nos negamos a extraer características
            if len(resultados.multi_hand_landmarks) > 1:
                return None 
                
            mano = resultados.multi_hand_landmarks[0]
            puntos = np.array([[punto.x, punto.y] for punto in mano.landmark])
            return puntos
            
        return None

    def compare(self, features_actuales):
        if not self.models or features_actuales is None:
            return "Sin modelos (o manos mezcladas)", float('inf'), []

        resultados = []
        for nombre, features_modelo in self.models.items():
            try:
                _, _, disparidad = procrustes(features_modelo, features_actuales)
                confianza = max(0.0, (1.0 - (disparidad / 0.2)) * 100.0)
                resultados.append((nombre, disparidad, confianza))
            except ValueError:
                continue

        if not resultados:
            return "Gesto desconocido", float('inf'), []

        resultados.sort(key=lambda x: x[1])
        
        mejor_nombre, mejor_distancia, _ = resultados[0] 

        # Lista para la pantalla
        ranking_pantalla = [f"{n}: Dist {d:.3f} (Conf {c:.1f}%)" for n, d, c in resultados]

        if mejor_distancia > 0.10:
            return "Gesto desconocido", mejor_distancia, ranking_pantalla

        return mejor_nombre, mejor_distancia, ranking_pantalla