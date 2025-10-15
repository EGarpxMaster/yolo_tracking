import cv2
from ultralytics import YOLO
import numpy as np
import time 
import argparse
import os
import glob
from coco_classes import filter_classes_by_category, get_class_name

# Eliminamos la limpieza de archivos PNG ya que no los vamos a generar
# files = glob.glob('output/*.png')
# for f in files:
#    os.remove(f)

# Global variables for line drawing interface
drawing_lines = []  # List of completed lines, each line is [(x1,y1), (x2,y2)]
current_line = []   # Current line being drawn (0, 1, or 2 points)
line_colors = [
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 0),     # Green
    (255, 128, 0),   # Orange
    (128, 0, 255),   # Purple
    (255, 255, 0),   # Cyan
    (0, 128, 255),   # Light blue
    (255, 0, 128),   # Pink
    (255, 255, 255), # White
    (0, 0, 255),     # Red
]

def draw_interface(frame, lines, current_line_points=None):
    """Draw the complete interface with instructions and lines"""
    display_frame = frame.copy()
    
    # Draw instructions
    instructions = [
        "Click 2 points to draw a counting line",
        "Press 'u' to undo last line",
        "Press 'r' to reset all lines",
        "Press ENTER when done"
    ]
    
    y_offset = 30
    for instruction in instructions:
        cv2.putText(display_frame, instruction, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    # Draw all completed lines
    for i, line in enumerate(lines):
        color = line_colors[i % len(line_colors)]
        cv2.line(display_frame, line[0], line[1], color, 3)
        cv2.circle(display_frame, line[0], 5, color, -1)
        cv2.circle(display_frame, line[1], 5, color, -1)
        
        # Add line number
        mid_x = (line[0][0] + line[1][0]) // 2
        mid_y = (line[0][1] + line[1][1]) // 2
        cv2.putText(display_frame, f"L{i+1}", (mid_x, mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw current line being drawn
    if current_line_points:
        if len(current_line_points) == 1:
            cv2.circle(display_frame, current_line_points[0], 5, (255, 255, 255), -1)
        elif len(current_line_points) == 2:
            cv2.line(display_frame, current_line_points[0], current_line_points[1], (255, 255, 255), 2)
            cv2.circle(display_frame, current_line_points[0], 5, (255, 255, 255), -1)
            cv2.circle(display_frame, current_line_points[1], 5, (255, 255, 255), -1)
    
    return display_frame

def setup_counting_lines_fixed(first_frame):
    """Fixed interactive interface to draw counting lines"""
    global drawing_lines, current_line
    
    drawing_lines = []
    current_line = []
    base_frame = first_frame.copy()
    
    window_name = "Setup Counting Lines"
    cv2.namedWindow(window_name)
    
    def mouse_callback(event, x, y, flags, param):
        global drawing_lines, current_line
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(current_line) == 0:
                # First point
                current_line = [(x, y)]
                print(f"[INFO] First point selected: ({x}, {y})")
            elif len(current_line) == 1:
                # Second point - complete the line
                current_line.append((x, y))
                drawing_lines.append(current_line)
                print(f"[INFO] Line {len(drawing_lines)} created: {current_line[0]} -> {current_line[1]}")
                current_line = []
            
            # Update display
            display_frame = draw_interface(base_frame, drawing_lines, current_line)
            cv2.imshow(window_name, display_frame)
    
    cv2.setMouseCallback(window_name, mouse_callback, base_frame)
    
    # Initial display
    display_frame = draw_interface(base_frame, drawing_lines, current_line)
    cv2.imshow(window_name, display_frame)
    
    print("\n" + "="*60)
    print("COUNTING LINES SETUP - INSTRUCTIONS")
    print("="*60)
    print("1. Click two points to create a counting line")
    print("2. You can create multiple lines")
    print("3. Press 'u' to undo the last line")
    print("4. Press 'r' to reset (clear all)")
    print("5. Press ENTER when done")
    print("6. Press 'q' or ESC to exit without saving")
    print("="*60 + "\n")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # ENTER
            if len(drawing_lines) > 0:
                print(f"[INFO] Setup complete! {len(drawing_lines)} line(s) configured.")
                cv2.destroyWindow(window_name)
                return drawing_lines
            else:
                print("[WARN] Please draw at least one counting line!")
        
        elif key == ord('u'):  # Undo
            if len(drawing_lines) > 0:
                removed = drawing_lines.pop()
                print(f"[INFO] Removed line: {removed}")
                current_line = []
                display_frame = draw_interface(base_frame, drawing_lines, current_line)
                cv2.imshow(window_name, display_frame)
            else:
                print("[WARN] No lines to undo!")
        
        elif key == ord('r'):  # Reset
            drawing_lines = []
            current_line = []
            print("[INFO] All lines cleared!")
            display_frame = draw_interface(base_frame, drawing_lines, current_line)
            cv2.imshow(window_name, display_frame)
        
        elif key == ord('q') or key == 27:  # q or ESC
            print("[INFO] Setup cancelled.")
            cv2.destroyWindow(window_name)
            exit()

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("--classes", type=str, default="vehicles", help="classes to detect: 'vehicles', 'people', 'people_and_vehicles', 'transportation', 'traffic', 'all'")
ap.add_argument("--cpu", action="store_true", help="force CPU usage even if GPU available")
ap.add_argument("--no-display", action="store_true", help="run without display (for headless systems)")
args = vars(ap.parse_args())

# Configuración inicial
print("[INFO] Cargando YOLOv11x con BoTSORT desde Ultralytics...")
model = YOLO('yolo11n.pt')

# Get selected classes for detection - Enfocado en vehículos
selected_classes = filter_classes_by_category("vehicles")
print(f"[INFO] Detectando clases: vehicles")
print(f"[INFO] IDs de Clases: {selected_classes}")
class_names = [get_class_name(i) for i in selected_classes]
print(f"[INFO] Nombres de Clases: {class_names}")

# Global variables for tracking and counting
memory = {}
counter = 0
line_counts = {}
counted_ids_per_line = {}

# Inicializar el video
cap = cv2.VideoCapture(args["input"])
if not cap.isOpened():
    print(f"[ERROR] No se pudo abrir el archivo de video: {args['input']}")
    exit()

# Read first frame to setup counting lines
print("[INFO] Leyendo primer frame para configuración de líneas...")
ret, first_frame = cap.read()
if not ret:
    print("[ERROR] ¡No se pudo leer el primer frame!")
    cap.release()
    exit()

# Setup counting lines with interactive interface - USAR LA VERSIÓN CORREGIDA
counting_lines = setup_counting_lines_fixed(first_frame)

# Reset video to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Initialize per-line tracking structures
for i in range(len(counting_lines)):
    line_counts[i] = {}
    counted_ids_per_line[i] = set()

# Configurar video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = None

frameIndex = 0
start_time = time.time()

# try to determine the total number of frames in the video file
try:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {total_frames} frames totales en el video")
except:
    total_frames = -1
    print("[INFO] No se pudo determinar el número total de frames")

print("\n[INFO] Iniciando procesamiento del video...")

while True:
    success, frame = cap.read()
    if not success:
        break
        
    # Get frame dimensions
    if frameIndex == 0:
        H, W = frame.shape[:2]
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

    # Detección y seguimiento con YOLO + BoTSORT
    results = model.track(
        frame, 
        conf=args["confidence"], 
        classes=selected_classes,
        tracker="bytetrack.yaml",
        verbose=False,
        persist=True,
        device='cpu' if args["cpu"] else None
    )

    # Process tracking results
    boxes = []
    track_ids = []
    class_ids = []
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        for box, track_id, class_id in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.cls):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            track_id = int(track_id.cpu().numpy())
            class_id = int(class_id.cpu().numpy())
            
            boxes.append([x1, y1, x2, y2])
            track_ids.append(track_id)
            class_ids.append(class_id)

    # Process each detection
    for i, (box, track_id, class_id) in enumerate(zip(boxes, track_ids, class_ids)):
        x1, y1, x2, y2 = box
        
        # Calculate center point
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Draw bounding box
        color = [int(c) for c in np.random.randint(0, 255, 3)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID and class
        class_name = get_class_name(class_id)
        text = f"ID:{track_id} {class_name}"
        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Check for line crossings
        if track_id in memory:
            prev_cx, prev_cy = memory[track_id]
            
            # Draw trajectory line
            cv2.line(frame, (prev_cx, prev_cy), (cx, cy), color, 2)
            
            # Check intersection with each counting line
            for line_idx, counting_line in enumerate(counting_lines):
                if intersect((prev_cx, prev_cy), (cx, cy), counting_line[0], counting_line[1]):
                    # Only count this track ID once per line
                    if track_id not in counted_ids_per_line[line_idx]:
                        counter += 1
                        counted_ids_per_line[line_idx].add(track_id)
                        
                        # Increment per-line counter
                        if class_name not in line_counts[line_idx]:
                            line_counts[line_idx][class_name] = 0
                        line_counts[line_idx][class_name] += 1
                        
                        print(f"✅ Vehículo {track_id} ({class_name}) contado en línea {line_idx+1}. Total: {counter}")
        
        # Update memory with current position
        memory[track_id] = (cx, cy)

    # Draw all counting lines with their colors and labels
    for line_idx, counting_line in enumerate(counting_lines):
        color = line_colors[line_idx % len(line_colors)]
        cv2.line(frame, counting_line[0], counting_line[1], color, 3)
        cv2.circle(frame, counting_line[0], 5, color, -1)
        cv2.circle(frame, counting_line[1], 5, color, -1)
        
        # Draw line number
        mid_x = (counting_line[0][0] + counting_line[1][0]) // 2
        mid_y = (counting_line[0][1] + counting_line[1][1]) // 2
        cv2.putText(frame, f"L{line_idx+1}", (mid_x, mid_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Draw total counter (top center)
    cv2.rectangle(frame, (W//2 - 120, 20), (W//2 + 120, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"TOTAL: {counter}", (W//2 - 100, 50), 
               cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)

    # Draw per-line counts on top-right corner
    start_y = 30
    panel_width = 220
    cv2.rectangle(frame, (W - panel_width - 5, 5), (W - 5, 30 + len(counting_lines)*60), (0, 0, 0), -1)
    for line_idx in range(len(counting_lines)):
        color = line_colors[line_idx % len(line_colors)]
        cv2.putText(frame, f"Linea {line_idx+1}:", (W - panel_width, start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        start_y += 20
        
        line_total = sum(line_counts[line_idx].values()) if line_counts[line_idx] else 0
        cv2.putText(frame, f"  Total: {line_total}", (W - panel_width, start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        start_y += 18
        
        for cls_name, cnt in line_counts[line_idx].items():
            text = f"  {cls_name}: {cnt}"
            cv2.putText(frame, text, (W - panel_width, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            start_y += 15
        start_y += 8

    # ELIMINADO: Guardar cada frame como imagen
    # cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)
    
    # Solo mostrar el frame si no estamos en modo sin display
    if not args["no_display"]:
        cv2.imshow("Video Processing", frame)
        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Procesamiento interrumpido por el usuario")
            break

    # Escribir frame al video de salida
    writer.write(frame)

    # Increase frame index
    frameIndex += 1

    # Optional: limit frames for testing
    if frameIndex >= 4000:
        print("[INFO] Límite de frames alcanzado, terminando...")
        break

# Calculate performance metrics
end_time = time.time()
total_time = end_time - start_time
fps = frameIndex / total_time if total_time > 0 else 0

# Release resources
cap.release()
if writer is not None:
    writer.release()

# Cerrar ventanas de visualización
if not args["no_display"]:
    cv2.destroyAllWindows()

# Print final results
print("\n" + "="*60)
print("RESULTADOS FINALES DEL CONTEO")
print("="*60)
print(f"Tiempo total de procesamiento: {total_time:.2f} segundos")
print(f"FPS promedio: {fps:.2f}")
print(f"Total general de vehículos: {counter}")

print("\nTotales por línea:")
for line_idx in range(len(counting_lines)):
    line_total = sum(line_counts[line_idx].values()) if line_counts[line_idx] else 0
    print(f"  Línea {line_idx+1}: {line_total} total")
    for cls_name, cnt in line_counts[line_idx].items():
        print(f"    {cls_name}: {cnt}")

# Save counts to file
try:
    os.makedirs('output', exist_ok=True)
    with open('output/counts.txt', 'w', encoding='utf-8') as f:
        f.write(f"RESULTADOS DEL CONTEO DE VEHÍCULOS\n")
        f.write(f"====================================\n")
        f.write(f"Tiempo total: {total_time:.2f} segundos\n")
        f.write(f"FPS promedio: {fps:.2f}\n")
        f.write(f"TOTAL_GENERAL:{counter}\n\n")
        
        f.write("TOTALES_POR_LINEA:\n")
        for line_idx in range(len(counting_lines)):
            line_total = sum(line_counts[line_idx].values()) if line_counts[line_idx] else 0
            f.write(f"\nLinea_{line_idx+1}_total:{line_total}\n")
            for cls_name, cnt in line_counts[line_idx].items():
                f.write(f"Linea_{line_idx+1}_{cls_name}:{cnt}\n")
    
    print(f"\n[INFO] Resultados guardados en output/counts.txt")
    print("="*60 + "\n")
except Exception as e:
    print(f"[WARN] No se pudieron guardar los conteos: {e}")