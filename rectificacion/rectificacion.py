#!/usr/bin/env python
import sys
import cv2
import numpy as np

def cargar_puntos(ruta):
    modo = "eder"
    pts_img, pts_real = [], []
    pt_obj1, pt_obj2 = None, None
    
    with open(ruta) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("# MODO"):
                modo = line.split()[2].lower()
                continue
            if line.startswith("#"): continue
            
            tokens = line.split()
            if tokens[0] == "REF":
                pts_img.append([float(tokens[1]), float(tokens[2])])
                pts_real.append([float(tokens[3]), float(tokens[4])])
            elif tokens[0] == "OBJ1":
                pt_obj1 = np.array([float(tokens[1]), float(tokens[2])], dtype=np.float32)
            elif tokens[0] == "OBJ2":
                pt_obj2 = np.array([float(tokens[1]), float(tokens[2])], dtype=np.float32)
                
    return modo, np.array(pts_img, dtype=np.float32), np.array(pts_real, dtype=np.float32), pt_obj1, pt_obj2

def main():
    archivo_txt = "puntos_referencia.txt"
    try:
        modo, pts_img, pts_real, pt_obj1, pt_obj2 = cargar_puntos(archivo_txt)
    except FileNotFoundError:
        sys.exit(f"No se encuentra {archivo_txt}.")

    img_path = sys.argv[1] if len(sys.argv) > 1 else ("gol-eder.png" if modo == "eder" else "mi_mesa.png")
    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f"No se pudo leer la imagen {img_path}")

    # Calcular Homografía
    H, _ = cv2.findHomography(pts_img, pts_real)
    out = img.copy()

    if modo == "eder":
        centro_porteria_real = np.array([0.0, 20.16], dtype=np.float32)
        src_eder = np.array([[pt_obj1]], dtype=np.float32)
        eder_real = cv2.perspectiveTransform(src_eder, H)[0, 0]
        distancia = np.linalg.norm(eder_real - centro_porteria_real)

        eder_ruido = cv2.perspectiveTransform(np.array([[[pt_obj1[0]+2, pt_obj1[1]+2]]], dtype=np.float32), H)[0, 0]
        margen_error = abs(distancia - np.linalg.norm(eder_ruido - centro_porteria_real))

        H_inv = np.linalg.inv(H)
        src_porteria = np.array([[[centro_porteria_real[0], centro_porteria_real[1]]]], dtype=np.float32)
        pt_porteria_img = cv2.perspectiveTransform(src_porteria, H_inv)[0, 0]

        p_orig, p_dest = tuple(pt_obj1.astype(int)), tuple(pt_porteria_img.astype(int))
        unidad = "m"
        texto_origen, texto_dest = "Eder", "Porteria"
        

        escala = 15
        margen_x, margen_y = 10 * escala, 10 * escala
        ancho_lienzo, alto_lienzo = int(40 * escala), int(60 * escala)
        p_dest_vista = (int(margen_x), int((20.16 * escala) + margen_y))
        
    else: 

        src_objs = np.array([[pt_obj1], [pt_obj2]], dtype=np.float32)
        objs_real = cv2.perspectiveTransform(src_objs, H)
        distancia = np.linalg.norm(objs_real[0, 0] - objs_real[1, 0])


        objs_ruido = cv2.perspectiveTransform(np.array([[[pt_obj1[0]+2, pt_obj1[1]+2]], [[pt_obj2[0]-2, pt_obj2[1]-2]]], dtype=np.float32), H)
        margen_error = abs(distancia - np.linalg.norm(objs_ruido[0, 0] - objs_ruido[1, 0]))

        p_orig, p_dest = tuple(pt_obj1.astype(int)), tuple(pt_obj2.astype(int))
        unidad = "cm"
        texto_origen, texto_dest = "Obj A", "Obj B"
        src_eder = src_objs

        escala = 10
        margen_x, margen_y = 5 * escala, 5 * escala
        ancho_lienzo, alto_lienzo = int(40 * escala), int(35 * escala)

    poly = pts_img.astype(int).reshape(-1, 1, 2)
    cv2.polylines(out, [poly], True, (0, 255, 0), 2)
    for i, (x, y) in enumerate(pts_img.astype(int)):
        cv2.circle(out, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(out, f"P{i+1}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.line(out, p_orig, p_dest, (0, 255, 255), 2) 
    
    cv2.circle(out, p_orig, 6, (0, 0, 255), -1) 
    cv2.putText(out, texto_origen, (p_orig[0] + 8, p_orig[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    color_dest = (255, 0, 255) if modo == "eder" else (0, 0, 255)
    cv2.circle(out, p_dest, 6, color_dest, -1) 
    cv2.putText(out, texto_dest, (p_dest[0] - 65 if modo == "eder" else p_dest[0] + 8, p_dest[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_dest, 2)

    texto = f"Distancia: {distancia:.2f} {unidad} (+/- {margen_error:.2f} {unidad})"
    cv2.rectangle(out, (5, 5), (420, 40), (0, 0, 0), -1)
    cv2.putText(out, texto, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(f"resultado_{modo}.png", out)
    print(f"\n Distancia exacta: {distancia:.2f} {unidad}")
    cv2.imshow("Resultado final", out)


    pts_real_px = (pts_real * escala) + np.array([margen_x, margen_y], dtype=np.float32)
    H_vista, _ = cv2.findHomography(pts_img, pts_real_px)
    imagen_rectificada = cv2.warpPerspective(img, H_vista, (ancho_lienzo, alto_lienzo))
    
    if modo == "eder":
        eder_vista = cv2.perspectiveTransform(src_eder, H_vista)[0, 0]
        p_orig_vista = (int(eder_vista[0]), int(eder_vista[1]))
    else:
        objs_vista = cv2.perspectiveTransform(src_eder, H_vista)
        p_orig_vista = (int(objs_vista[0, 0][0]), int(objs_vista[0, 0][1]))
        p_dest_vista = (int(objs_vista[1, 0][0]), int(objs_vista[1, 0][1]))
        
    cv2.circle(imagen_rectificada, p_orig_vista, 6, (0, 0, 255), -1)
    cv2.circle(imagen_rectificada, p_dest_vista, 6, color_dest, -1)
    cv2.line(imagen_rectificada, p_orig_vista, p_dest_vista, (0, 255, 255), 2)
    
    cv2.imshow("Vista rectificada", imagen_rectificada)
    cv2.imwrite(f"vista_rectificada_{modo}.png", imagen_rectificada)
    print("Imágenes guardadas correctamente.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()