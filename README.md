# Visión Artificial — Computer Vision Toolkit

> Colección de módulos de visión por computador desarrollados en Python: detección de objetos en tiempo real, calibración de cámara, clasificación por múltiples métodos, realidad aumentada y análisis de tráfico.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-v8%2Fv11-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-purple)
![TFLite](https://img.shields.io/badge/TensorFlow_Lite-✓-yellow?logo=tensorflow)

---

## Módulos

| Módulo | Descripción | Técnicas principales |
|---|---|---|
| [Detección de actividad](#detección-de-actividad) | Vigilancia inteligente con alertas Telegram | YOLOv8, MOG2, threading |
| [Clasificador modular](#clasificador-modular) | Reconocimiento de objetos por múltiples métodos | SIFT, embeddings, Procrustes |
| [Calibración de cámara](#calibración-de-cámara) | Medición 3D y corrección de distorsión | Matriz intrínseca, FOV, undistort |
| [Rectificación perspectiva](#rectificación-de-perspectiva) | Medición de distancias reales en imagen | Homografía, transformación métrica |
| [Entrenamiento YOLO](#entrenamiento-yolo-personalizado) | Detector de objetos custom entrenado desde cero | YOLOv8/11, dataset propio |
| [AR con marcadores](#ar-con-marcadores-hexagonales) | Realidad aumentada sobre marcadores hexagonales | solvePnP, Rodrigues, pose estimation |
| [Sustitución en carnet](#sustitución-de-foto-en-carnet) | Reemplaza la foto de un carnet en tiempo real | ORB, RANSAC, alpha blending |

---

## Detección de actividad

**Archivo:** [`actividad/actividad_via.py`](actividad/actividad_via.py)

Sistema de vigilancia que detecta personas en zonas definidas por el usuario y envía alertas a Telegram con vídeo adjunto.

**Características:**
- Detección con **YOLOv8n** (inferencia en tiempo real).
- **Pre-buffering de 3 s** antes del evento y post-buffering de 2 s, garantizando que el clip captura la acción completa.
- **Difuminado de rostros** (Gaussian blur) antes de enviar el vídeo — privacidad por diseño.
- Notificaciones asíncronas mediante **threading** para no bloquear el bucle de captura.
- Sustracción de fondo **MOG2** + análisis de contornos como detección de movimiento previo a YOLO.
- ROI configurable interactivamente con callbacks de ratón.

**Configuración del bot:**
```bash
# Crear actividad/token.env con tus propias credenciales
TOKEN = "tu_token_de_botfather"
USER_ID = "tu_chat_id"
```

---

## Clasificador modular

**Archivos:** [`clasificador/`](clasificador/)

Framework de clasificación de imágenes que carga métodos dinámicamente sin cambiar el código principal. Compara la entrada de la cámara contra modelos de referencia guardados.

**Métodos disponibles:**

| Método | Archivo | Algoritmo | Mejor para |
|---|---|---|---|
| SIFT | [`metodos/SIFT.py`](clasificador/metodos/SIFT.py) | FLANN KD-tree + ratio test de Lowe (0.7) | Objetos con textura rica, cambios de escala/rotación |
| Embeddings | [`metodos/mediapipe_embedding.py`](clasificador/metodos/mediapipe_embedding.py) | MobileNetV3 + similitud coseno | Objetos con diferencias sutiles |
| Procrustes | [`metodos/procrustes.py`](clasificador/metodos/procrustes.py) | MediaPipe Hands + análisis de Procrustes | Reconocimiento de gestos de mano |

**Uso:**
```bash
# Ejecutar con un método concreto
python clasificador/main.py --method SIFT
python clasificador/main.py --method mediapipe_embedding
python clasificador/main.py --method procrustes
```

---

## Calibración de cámara

**Archivos:** [`calibracion/`](calibracion/)

Herramienta interactiva para calibración intrínseca de cámara y medición 3D en escena.

- Calcula la **matriz intrínseca K** y coeficientes de distorsión **D** a partir de capturas del patrón de calibración.
- Interfaz con **sliders** para ajustar focal (f), distancia Z y altura de objetos en tiempo real.
- Proyección perspectiva de **grids 3D** sobre la imagen corregida.
- Conversión pixel → coordenadas 3D mediante proyección inversa.
- Cálculo de **FOV** horizontal y vertical.

**Archivos de calibración incluidos:**
- [`calibracion/calib.txt`](calibracion/calib.txt) — parámetros K, D de la cámara usada.
- [`calibracion/calibrate/pattern.png`](calibracion/calibrate/pattern.png) — patrón de tablero de ajedrez.
- [`calibracion/calibrate/capturas-stream/`](calibracion/calibrate/capturas-stream/) — 16 capturas de calibración.

---

## Rectificación de perspectiva

**Archivos:** [`rectificacion/`](rectificacion/)

Estima distancias reales entre objetos usando homografía desde puntos de referencia conocidos.

- Soporta dos modos: medición en campo de fútbol y medición general de objetos.
- Genera la vista **cenital rectificada** y la imagen original anotada.
- Calcula el **margen de error** perturbando las coordenadas de referencia.

---

## Entrenamiento YOLO personalizado

**Archivos:** [`DL/`](DL/)

Entrenamiento de un detector de objetos custom (mando a distancia) con YOLOv8/11.

- Dataset propio: **200 imágenes de entrenamiento** + 34 de validación, anotadas en formato YOLO.
- Configuración en [`DL/mando.yaml`](DL/mando.yaml).
- Script de inferencia mínimo en [`DL/yolo_run.py`](DL/yolo_run.py).

**Pesos del modelo entrenado:**
Los ficheros `.pt` no se incluyen en el repositorio por su tamaño. Para usar el detector, entrena el modelo con el dataset incluido:

**Entrenamiento desde cero:**
```bash
yolo train model=yolo11n.pt data=DL/mando.yaml epochs=50 imgsz=640
```

---

## AR con marcadores hexagonales

**Archivos:** [`opcionales/realidad-virtual/`](opcionales/realidad-virtual/)

Aplicación de realidad aumentada que detecta un marcador hexagonal, estima su pose 3D y permite colocar cubos de alambre animados sobre él.

**Detalles técnicos:**
- Extracción del hexágono: umbralización de Otsu + operaciones morfológicas + aproximación de contornos.
- Estimación de pose con `solvePnP`, probando todas las rotaciones cíclicas de vértices para la correspondencia óptima.
- **Suavizado de pose**: media de historial de 5 frames en espacio de eje-ángulo (SO(3)) reconstruida con Rodrigues.
- Animación: interpolación exponencial (α = 0.08) hacia el punto objetivo.
- Deprojección de clics de ratón usando homografía para colocar objetos en el plano del marcador.
- HUD con FPS, estado del marcador y número de objetos activos.

---

## Sustitución de foto en carnet

**Archivos:** [`opcionales/sustitucion-foto-carnet/`](opcionales/sustitucion-foto-carnet/)

Reemplaza en tiempo real la fotografía de un carnet detectado en vídeo.

- Detección del carnet mediante **ORB** (2000 features) + matching Hamming + ratio test de Lowe (0.75).
- Estimación de homografía con **RANSAC** (umbral de reproyección 5 px).
- Composición final por warping perspectivo y **alpha blending** con máscara binaria.
- Versión alternativa con seguimiento **Lucas-Kanade** en [`sustitucion-carnet-LK.py`](opcionales/sustitucion-foto-carnet/sustitucion-carnet-LK.py).

---

## Stack tecnológico

| Librería | Uso |
|---|---|
| OpenCV 4.8+ | Captura, procesado, homografía, detección de features |
| Ultralytics YOLOv8/11 | Detección de objetos, entrenamiento |
| MediaPipe | Landmarks de mano, embeddings MobileNetV3 |
| TensorFlow Lite | Inferencia de embeddings |
| SciPy | Análisis de Procrustes |
| NumPy | Álgebra lineal y operaciones matriciales |
| python-dotenv | Gestión segura de credenciales |
| requests | Envío de notificaciones a Telegram Bot API |

---

## Instalación

```bash
git clone https://github.com/tu-usuario/Vision-Artificial.git
cd Vision-Artificial

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

**Modelos pre-entrenados:** YOLOv8n y los modelos de MediaPipe se descargan automáticamente en la primera ejecución. El modelo custom de `DL/` debe entrenarse con el dataset incluido (ver sección [Entrenamiento YOLO personalizado](#entrenamiento-yolo-personalizado)).

---

## Estructura del repositorio

```
Vision-Artificial/
├── actividad/              # Detección de actividad + alertas Telegram
├── calibracion/            # Calibración intrínseca y medición 3D
│   └── calibrate/          # Scripts de calibración + imágenes de patrón
├── clasificador/           # Clasificador modular
│   └── metodos/            # SIFT · embeddings · Procrustes
├── DL/                     # Dataset y entrenamiento YOLO custom
│   ├── train/              # 200 imágenes de entrenamiento + labels
│   └── val/                # 34 imágenes de validación + labels
├── opcionales/
│   ├── realidad-virtual/   # AR con marcadores hexagonales
│   └── sustitucion-foto-carnet/  # Reemplazo de foto en carnet
├── rectificacion/          # Rectificación perspectiva y medición
├── trafico/                # Captura y visualización de tráfico
├── requirements.txt
└── .gitignore
```

---

## Licencia

Este proyecto se distribuye bajo la licencia [MIT](LICENSE).
