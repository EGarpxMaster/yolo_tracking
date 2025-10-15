import numpy as np
from sklearn.metrics import mean_absolute_error

def evaluar_aforo(real, predicho):
    mae = mean_absolute_error(real, predicho)
    mape = np.mean(np.abs((real - predicho) / real)) * 100
    accuracy_porcentual = 100 - mape
    
    print(f"Exactitud aproximada: {accuracy_porcentual:.2f}%")
    print(f"Error promedio: {mae:.2f} vehículos")

    return accuracy_porcentual, mae

yolo11n_counts = np.array([13, 13, 55, 44, 50, 50])
yolo11s_counts = np.array([15, 15, 57, 57, 57, 57])
yolov8s_counts = np.array([14, 14, 61, 72, 59, 59])

real_counts = np.array([14, 14, 27 , 27, 46, 46])

yolo11n_time = np.array([317.20, 317.20, 54.89, 57.23, 307.42, 317.35])  # Tiempos en segundos
yolo11s_time = np.array([993.6108, 4990.6250, 54.89, 57.23, 541.33, 531.19])  # Tiempos en segundos
yolo8s_time = np.array([584.14, 584.14, 600.97, 57.23, 571.46, 576.93])  # Tiempos en segundos

real_time = np.array([60, 60, 60, 60, 60, 60])  # Tiempos en segundos



print("Evaluación para YOLOv11n:")
evaluar_aforo(real_counts, yolo11n_counts)
print("\nEvaluación para YOLOv11s:")
evaluar_aforo(real_counts, yolo11s_counts)
print("\nEvaluación para YOLOv8s:")
evaluar_aforo(real_counts, yolov8s_counts)