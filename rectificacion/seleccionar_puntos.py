#!/usr/bin/env python
import sys
import cv2
import numpy as np


PERFILES = {
    "eder": {
        "img_defecto": "gol-eder.png",
        "etiquetas": [
            "P1 (Fondo-lejano) -> (0, 0)",
            "P2 (Frontal-lejano) -> (16.5, 0)",
            "P3 (Frontal-cercano) -> (16.5, 40.32)",
            "P4 (PUNTO PENALTI) -> (11.0, 20.16)",
            "EDER (Pie de atrás)"
        ],
        "coords_reales": [
            (0.0, 0.0), (16.5, 0.0), (16.5, 40.32), (11.0, 20.16)
        ]
    },
    "folio": {
        "img_defecto": "folio.jpeg",
        "etiquetas": [
            "Folio (Sup-Izq) -> (0, 0)",
            "Folio (Sup-Der) -> (29.7, 0)",
            "Folio (Inf-Der) -> (29.7, 21.0)",
            "Folio (Inf-Izq) -> (0, 21.0)",
            "Haz clic en OBJETO A",
            "Haz clic en OBJETO B"
        ],
        "coords_reales": [
            (0.0, 0.0), (29.7, 0.0), (29.7, 21.0), (0.0, 21.0)
        ]
    }
}

def main():

    modo = sys.argv[1].lower() if len(sys.argv) > 1 and sys.argv[1].lower() in PERFILES else "eder"
    perfil = PERFILES[modo]
    
    path = sys.argv[2] if len(sys.argv) > 2 else perfil["img_defecto"]
    img0 = cv2.imread(path)
    if img0 is None:
        sys.exit(f"No se pudo leer {path}")

    print(f"--- Modo de captura: {modo.upper()} ---")


    alto_maximo = 800
    
    if img0.shape[0] > alto_maximo:
        escala = alto_maximo / img0.shape[0]
    else:
        escala = 2
        
    base = cv2.resize(img0, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    puntos = []
    total_clics = len(perfil["etiquetas"])

    def redibujar():
        vis = base.copy()
        idx = len(puntos)
        if idx < total_clics:
            cv2.putText(vis, f"Clic {idx+1}: {perfil['etiquetas'][idx]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(vis, "OK -> pulsa 'q' para guardar y salir", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
        for i, (x, y) in enumerate(puntos):
            X, Y = int(x * escala), int(y * escala)
            color = (0, 0, 255) if i >= 4 else (0, 255, 0)
            

            if i < 4: etiqueta = f"P{i+1}"
            elif modo == "eder": etiqueta = "Eder"
            else: etiqueta = "Obj A" if i == 4 else "Obj B"
            
            cv2.circle(vis, (X, Y), 6, color, -1)
            cv2.putText(vis, etiqueta, (X + 8, Y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("Seleccion de Puntos", vis)

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < total_clics:
            puntos.append((x / escala, y / escala))
            redibujar()

    cv2.namedWindow("Seleccion de Puntos")
    cv2.setMouseCallback("Seleccion de Puntos", click)
    redibujar()
    
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == ord("q"): break
        if k == ord("u") and puntos: puntos.pop(); redibujar()
        if k == ord("r"): puntos.clear(); redibujar()
            
    cv2.destroyAllWindows()

    if len(puntos) < total_clics:
        print("Faltan puntos por marcar. Abortando.")
        return


    with open("puntos_referencia.txt", "w") as f:
        f.write(f"# MODO {modo}\n")
        f.write("# img_x img_y real_x real_y\n")
        for (ix, iy), (rx, ry) in zip(puntos[:4], perfil["coords_reales"]):
            f.write(f"REF {ix:.2f} {iy:.2f} {rx:.2f} {ry:.2f}\n")
            
        f.write(f"OBJ1 {puntos[4][0]:.2f} {puntos[4][1]:.2f}\n")
        if modo == "folio":
            f.write(f"OBJ2 {puntos[5][0]:.2f} {puntos[5][1]:.2f}\n")
        
    print(f"\nArchivo 'puntos_referencia.txt' guardado correctamente (Modo: {modo}).")

if __name__ == "__main__":
    main()