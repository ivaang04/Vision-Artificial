#!/usr/bin/env python
from abc import ABC, abstractmethod

class IReconocedor(ABC):
    """
    Las clases para los métodos heredan de esta clase.
    """
    
    @abstractmethod
    def precompute(self, image):
        """
        Extrae las características matemáticas de la imagen.
        Debe devolver los datos extraídos o None si hay un error.
        """
        pass

    @abstractmethod
    def compare(self, features_actuales):
        """
        Compara las características con los modelos guardados.
        Debe devolver una tupla: (mejor_nombre, distancia)
        """
        pass