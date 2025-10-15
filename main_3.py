import cv2
from ultralytics import YOLO
import numpy as np
from sort import Sort
import time 

# Configuración inicial
model = YOLO('yolov8s.pt')  
tracker = Sort(max_age=20, min_hits=3)

# Clases de interés: coche, moto, autobús, camión
CLASS_IDS = [2, 3, 5, 7]

# Define la línea de conteo
LINE_POSITION = 600
crossed_ids = set()
vehicle_count = 0

cap = cv2.VideoCapture('/home/barba_negra/python-traffic-counter-with-yolo-and-sort/input/fkusamil_2_corto.mp4')  
time_0 = time.time()
while True:
    
    success, frame = cap.read()
    if not success:
        break
        
    # Detección con YOLO
    results = model.predict(frame, classes=CLASS_IDS, conf=0.3)  # Confidence más bajo para drones
    detections = np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections = np.vstack([detections, [x1, y1, x2, y2, conf]])
    
    # CORRECCIÓN: Manejar caso cuando no hay detecciones
    if detections.size == 0:
        detections = np.empty((0, 5))
    
    # Seguimiento con SORT
    track_results = tracker.update(detections)
    
    # CORRECCIÓN: Verificar que track_results no esté vacío
    if track_results.size > 0:
        # Asegurarnos de que track_results tenga la forma correcta
        if track_results.ndim == 1:
            track_results = track_results.reshape(1, -1)
            
        for res in track_results:
            # CORRECCIÓN: Manejar diferentes formas de resultados
            if len(res) >= 5:
                x1, y1, x2, y2, track_id = map(int, res[:5])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Lógica de conteo al cruzar la línea
                if LINE_POSITION - 10 <= cy <= LINE_POSITION + 10 and cx > 1100:
                    if track_id not in crossed_ids:
                        crossed_ids.add(track_id)
                        vehicle_count += 1
                        print(f"Vehículo {track_id} contado. Total: {vehicle_count}")
                        
                # Dibujar bounding box y ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Dibuja la línea y el contador
    cv2.line(frame, (0, LINE_POSITION), (1100, LINE_POSITION), (0, 0, 255), 3)
    cv2.putText(frame, f'Total: {vehicle_count}', (50, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    # Mostrar frame
    cv2.imshow('Conteo Vehicular', frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
time_1 = time.time()
print(f"Conteo final: {vehicle_count} vehículos")
print(f"Tiempo total de procesamiento: {time_1 - time_0:.2f} segundos")