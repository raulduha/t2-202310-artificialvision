#Raul Duhalde Errazuriz
#Tarea 2
import cv2
import numpy as np
from time import perf_counter

#abrir el archivo de video
cap = cv2.VideoCapture('visionartificial1.mp4')

#obtener los cuadros por segundo y el número total de cuadros
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#definir la constante para el filtro de la variable de tiempo
c = 0.05

#definir el códec y crear un objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('filtered_visionartificial1.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

#definir parámetros de segmentación
segment_count = fps * 3
scale_fact = 1
segment_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_fact / segment_count)
frames = []

#comenzar a procesar cada cuadro
t1 = perf_counter()
while(cap.isOpened()):
    ret, new_frame = cap.read()

    if new_frame is None:
        break

#escalar el cuadro si es necesario
    if scale_fact != 1:
        new_frame = cv2.resize(new_frame, (int(new_frame.shape[1]*scale_fact),int(new_frame.shape[0]*scale_fact)))

#  agregar el cuadro a la lista de cuadros
    frames.append(new_frame)

# aplicar la lógica de segmentación y concatenación una vez que tenemos suficientecuadros
    if len(frames) >= segment_count:    
        segments = []
        for i, frame in enumerate(frames):
            segments.append(frame[i*segment_height:(i+1)*segment_height])

        after_frame = np.concatenate(segments, axis=0)

# Aplicar el filtro de la variable de tiempo
        t = i / fps
        x = after_frame.shape[0] - 1
        t_new = t - c * x
# cambio de variable en el tiempo del tipo t′ = t − c · x donde t es el tiempo original, 
# c es la constante del filtro y x es la posición en la dirección vertical del cuadro segmentado. 
# matriz de transformación para aplicar el cambio de variable en el tiempo
        M = np.float32([[1, 0, 0], [0, 1, 0]])
        M[0, 2] = -c * t_new * fps
    
# aplicar la transformación a través de la función warpAffine de OpenCV
        filtered_frame = cv2.warpAffine(after_frame, M, (after_frame.shape[1], after_frame.shape[0]))

#escribir el cuadro procesado en el archivo de video de salida
        out.write(filtered_frame)
# mostrar el cuadro procesado
        cv2.imshow('frame', filtered_frame)

        # Esperar a que se presione la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#actualizar el tiempo y eliminar el cuadro más antiguo
        t1 = perf_counter()
        frames.pop(0)

cap.release()
out.release()
cv2.destroyAllWindows()
