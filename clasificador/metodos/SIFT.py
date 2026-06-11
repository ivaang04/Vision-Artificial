#!/usr/bin/env python
from . import IReconocedor
import cv2 as cv

class Reconocedor(IReconocedor):
    def __init__(self):
        self.models = {}
        self.sift = cv.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv.FlannBasedMatcher(index_params, search_params)

    def precompute(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        keypoints, descriptores = self.sift.detectAndCompute(gray, None)
        
        if descriptores is not None and len(descriptores) > 5:
            return descriptores
        return None

    def compare(self, features_actuales):
        if not self.models or features_actuales is None or len(features_actuales) < 2:
            return "Sin modelos", float('inf'), []

        resultados = []
        for nombre, features_modelo in self.models.items():
            matches = self.matcher.knnMatch(features_actuales, features_modelo, k=2)
            
            good_matches = 0
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.7 * n.distance:
                        good_matches += 1

            distancia = 100.0 / (good_matches + 1)
            confianza = min(100.0, (good_matches / 40.0) * 100.0) 
            
            resultados.append((nombre, distancia, confianza, good_matches))

        resultados.sort(key=lambda x: x[3], reverse=True)

        mejor_nombre, mejor_distancia, _, max_matches = resultados[0]

        # Lista para la pantalla
        ranking_pantalla = [f"{n}: {m} pts (Conf {c:.1f}%)" for n, d, c, m in resultados]

        if max_matches < 60:
            return "Desconocido", mejor_distancia, ranking_pantalla

        return mejor_nombre, mejor_distancia, ranking_pantalla