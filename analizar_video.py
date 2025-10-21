# import the necessary packages
import sys
import os
import numpy as np
import argparse
import time
import cv2
import glob
from ultralytics import YOLO
from coco_classes import filter_classes_by_category, get_class_name
import torch

# Obtener directorio actual y configurar rutas relativas
current_dir = os.path.dirname(os.path.abspath(__file__))
salida_path = os.path.join(current_dir, "output", "")  # Usar ruta relativa con barra final
# Crear directorio de salida si no existe
os.makedirs(salida_path, exist_ok=True)

confidence = 0.2
threshold = 0.3
classes = 'people_and_vehicles'
force_cpu= True # gpu = False, cpu = True

# GPU Configuration
def detect_device(force_cpu):
    """Detect and configure the best available device (GPU or CPU)"""
    if force_cpu:
        #print("[INFO] Forzando uso de CPU (--cpu flag)")
        return 'cpu'
    
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        #print(f"[INFO] 🚀 GPU detectada: {gpu_name}")
        #print(f"[INFO] 💾 Memoria GPU: {gpu_memory:.2f} GB")
        #print(f"[INFO] ⚡ CUDA Version: {torch.version.cuda}")
        print(f"[INFO] ✓ Usando GPU para aceleración máxima")
        return device
    else:
        print("[INFO] GPU no disponible, usando CPU")
        #print("[WARN] El procesamiento será más lento. Considera instalar PyTorch con CUDA:")
        #print("[WARN] pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return 'cpu'

# Global variables for line drawing interface
drawing_lines = []  # List of completed lines, each line is [(x1,y1), (x2,y2)]
current_line_point = None  # First point of the line being drawn
temp_frame = None  # Frame for drawing
line_colors = [
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 0),     # Green
    (255, 128, 0),   # Orange
    (128, 0, 255),   # Purple
    (255, 255, 0),   # Cyan
    (0, 128, 255),   # Light blue
    (255, 0, 128),   # Pink
]

# Mouse callback for drawing lines
def mouse_callback(event, x, y, flags, param):
    global current_line_point, temp_frame, drawing_lines
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_line_point is None:
            # First point of the line
            current_line_point = (x, y)
            #print(f"[INFO] First point selected: ({x}, {y}). Click again to complete the line.")
        else:
            # Second point - complete the line
            drawing_lines.append([current_line_point, (x, y)])
            #print(f"[INFO] Line {len(drawing_lines)} created: {current_line_point} -> ({x}, {y})")
            current_line_point = None
            
            # Redraw all lines
            temp_frame = param.copy()
            for i, line in enumerate(drawing_lines):
                color = line_colors[i % len(line_colors)]
                cv2.line(temp_frame, line[0], line[1], color, 3)
                cv2.circle(temp_frame, line[0], 5, color, -1)
                cv2.circle(temp_frame, line[1], 5, color, -1)
                # Add line number
                mid_x = (line[0][0] + line[1][0]) // 2
                mid_y = (line[0][1] + line[1][1]) // 2
                cv2.putText(temp_frame, f"L{i+1}", (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Setup Counting Lines", temp_frame)
    
    elif event == cv2.EVENT_MOUSEMOVE and current_line_point is not None:
        # Show preview of the line being drawn
        preview = temp_frame.copy()
        cv2.line(preview, current_line_point, (x, y), (255, 255, 255), 2)
        cv2.circle(preview, current_line_point, 5, (255, 255, 255), -1)
        cv2.imshow("Setup Counting Lines", preview)


def setup_counting_lines(first_frame):
    """Interactive interface to draw counting lines"""
    global temp_frame, drawing_lines, current_line_point
    
    temp_frame = first_frame.copy()
    drawing_lines = []
    current_line_point = None
    
    # Create window and set mouse callback
    cv2.namedWindow("Setup Counting Lines")
    cv2.setMouseCallback("Setup Counting Lines", mouse_callback, first_frame)
    
    # Draw instructions on frame
    instructions = [
        "Click 2 points to draw a counting line",
        "Press 'u' to undo last line",
        "Press 'r' to reset all lines",
        "Press ENTER when done"
    ]
    y_offset = 30
    for instruction in instructions:
        cv2.putText(temp_frame, instruction, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
    
    cv2.imshow("Setup Counting Lines", temp_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13 or key == 32:  # ENTER or SPACE
            if len(drawing_lines) > 0:
                #print(f"[INFO] Setup complete! {len(drawing_lines)} line(s) configured.")
                cv2.destroyWindow("Setup Counting Lines")
                return drawing_lines
            else:
                print("[WARN] Please draw at least one counting line!")
        
        elif key == ord('u') or key == ord('U'):  # Undo
            if len(drawing_lines) > 0:
                removed = drawing_lines.pop()
                #print(f"[INFO] Removed line: {removed}")
                # Redraw
                temp_frame = first_frame.copy()
                for i, line in enumerate(drawing_lines):
                    color = line_colors[i % len(line_colors)]
                    cv2.line(temp_frame, line[0], line[1], color, 3)
                    cv2.circle(temp_frame, line[0], 5, color, -1)
                    cv2.circle(temp_frame, line[1], 5, color, -1)
                    mid_x = (line[0][0] + line[1][0]) // 2
                    mid_y = (line[0][1] + line[1][1]) // 2
                    cv2.putText(temp_frame, f"L{i+1}", (mid_x, mid_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # Redraw instructions
                y_offset = 30
                for instruction in instructions:
                    cv2.putText(temp_frame, instruction, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25
                cv2.imshow("Setup Counting Lines", temp_frame)
            else:
                print("[WARN] No lines to undo!")
        
        elif key == ord('r') or key == ord('R'):  # Reset
            drawing_lines = []
            current_line_point = None
            temp_frame = first_frame.copy()
            # Redraw instructions
            y_offset = 30
            for instruction in instructions:
                cv2.putText(temp_frame, instruction, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            cv2.imshow("Setup Counting Lines", temp_frame)
            #print("[INFO] All lines cleared!")
        
        elif key == ord('q') or key == ord('Q') or key == 27:  # q or ESC
            #print("[INFO] Setup cancelled.")
            cv2.destroyWindow("Setup Counting Lines")
            exit()

# Global variables for tracking and counting
memory = {}
# Total counter (all objects)
counter = 0
# Per-line counters: {line_index: {class_name: count}}
line_counts = {}
# Per-class counters (total across all lines)
class_counts = {}
# Keep track of which track IDs have been counted per line: {line_index: set(track_ids)}
counted_ids_per_line = {}


# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Detect and configure device (GPU or CPU)
DEVICE = detect_device(force_cpu)

# Load YOLOv11x model with BoTSORT tracking
#print("[INFO] loading YOLOv11x with BoTSORT from Ultralytics...")
#model = YOLO('yolo11x.pt')  # This will automatically download the model if not present
model = YOLO('yolo11x.pt')

def analizar(video_path, nombre):
    global vs, writer, W, H, frameIndex, counter, line_counts, class_counts, counted_ids_per_line, memory

    # Set input and output paths
    output = salida_path + nombre + "_analizado.mp4"
    
    # Move model to GPU if available
    model.to(DEVICE)
    #print(f"[INFO] ✓ Modelo cargado en: {DEVICE.upper()}")

    # Enable optimizations for GPU
    if DEVICE == 'cuda':
        #print("[INFO] ⚡ Habilitando optimizaciones de GPU (FP16/half precision)")
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes

    # Get selected classes for detection
    selected_classes = filter_classes_by_category(classes)
    #print(f"[INFO] Detecting classes: {classes}")
    #print(f"[INFO] Class IDs: {selected_classes}")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    # initialize the video stream, pointer to output video file, and frame dimensions
    vs = cv2.VideoCapture(video_path)
    if not vs.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        exit()

    writer = None
    (W, H) = (None, None)

    frameIndex = 0

    # try to determine the total number of frames in the video file
    try:
        total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            #print("[INFO] Could not determine frame count, processing until end of video")
            total = -1
        else:
            print(f"[INFO] {total} total frames in video")
    except:
        #print("[INFO] could not determine # of frames in video")
        #print("[INFO] no approx. completion time can be provided")
        total = -1

    # Read first frame to setup counting lines
    #print("[INFO] Reading first frame for line setup...")
    ret, first_frame = vs.read()
    if not ret:
        #print("[ERROR] Could not read first frame!")
        vs.release()
        exit()

    # Setup counting lines with interactive interface
    counting_lines = setup_counting_lines(first_frame)
    
    # Reset video to beginning
    vs.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize per-line tracking structures
    for i in range(len(counting_lines)):
        line_counts[i] = {}
        counted_ids_per_line[i] = set()

    # Reset counters for new video
    counter = 0
    class_counts = {}
    memory = {}

    # Performance tracking
    total_processing_time = 0
    frame_times = []
    processing_start_time = time.time()

    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # Run YOLOv11x inference with BoTSORT tracking on GPU
        start = time.time()
        results = model.track(frame, conf=confidence, iou=threshold, 
                            tracker="botsort.yaml", verbose=False, classes=selected_classes,
                            persist=True, device=DEVICE, half=(DEVICE == 'cuda'))
        end = time.time()
        
        # Track performance
        frame_time = end - start
        frame_times.append(frame_time)
        total_processing_time += frame_time

        # Process tracking results
        boxes = []
        indexIDs = []
        classIDs = []
        previous = memory.copy()
        memory = {}

        # Extract tracking information from results
        if results[0].boxes is not None and results[0].boxes.id is not None:
            for i, (box, track_id, class_id) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.cls)):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                track_id = int(track_id.cpu().numpy())
                class_id = int(class_id.cpu().numpy())
                
                boxes.append([x1, y1, x2, y2])
                indexIDs.append(track_id)
                classIDs.append(class_id)
                memory[track_id] = [x1, y1, x2, y2]

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates (x1, y1, x2, y2)
                (x1, y1) = (int(box[0]), int(box[1]))
                (x2, y2) = (int(box[2]), int(box[3]))

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (px1, py1) = (int(previous_box[0]), int(previous_box[1]))
                    (px2, py2) = (int(previous_box[2]), int(previous_box[3]))
                    
                    # Calculate center points for trajectory
                    p0 = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    p1 = (int((px1 + px2) / 2), int((py1 + py2) / 2))
                    cv2.line(frame, p0, p1, color, 3)

                    # Check intersection with each counting line
                    for line_idx, counting_line in enumerate(counting_lines):
                        if intersect(p0, p1, counting_line[0], counting_line[1]):
                            # Only count this track ID once per line
                            if indexIDs[i] not in counted_ids_per_line[line_idx]:
                                counter += 1
                                counted_ids_per_line[line_idx].add(indexIDs[i])
                                
                                # Increment per-line and per-class counter
                                cls_name = get_class_name(classIDs[i])
                                if cls_name not in line_counts[line_idx]:
                                    line_counts[line_idx][cls_name] = 0
                                line_counts[line_idx][cls_name] += 1
                                
                                # Increment global per-class counter
                                if cls_name not in class_counts:
                                    class_counts[cls_name] = 0
                                class_counts[cls_name] += 1

                # Draw object ID and class label (always show class name)
                class_name = get_class_name(classIDs[i])
                text = f"ID:{indexIDs[i]} {class_name}"
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1

        # Draw all counting lines with their colors and labels
        for line_idx, counting_line in enumerate(counting_lines):
            color = line_colors[line_idx % len(line_colors)]
            cv2.line(frame, counting_line[0], counting_line[1], color, 5)
            
            # Draw line number near the line
            mid_x = (counting_line[0][0] + counting_line[1][0]) // 2
            mid_y = (counting_line[0][1] + counting_line[1][1]) // 2
            cv2.putText(frame, f"L{line_idx+1}", (mid_x, mid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw total counter (top center)
        cv2.putText(frame, f"Total: {counter}", (W//2 - 100, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)

        # Draw per-class totals on top-left corner
        start_y = 30
        cv2.putText(frame, "TOTALES:", (10, start_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        start_y += 30
        for cls_name, cnt in class_counts.items():
            text = f"{cls_name}: {cnt}"
            cv2.putText(frame, text, (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            start_y += 25

        # Draw per-line counts on top-right corner
        start_y = 30
        for line_idx in range(len(counting_lines)):
            color = line_colors[line_idx % len(line_colors)]
            cv2.putText(frame, f"Linea {line_idx+1}:", (W - 200, start_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            start_y += 25
            
            line_total = sum(line_counts[line_idx].values()) if line_counts[line_idx] else 0
            cv2.putText(frame, f"  Total: {line_total}", (W - 200, start_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            start_y += 20
            
            for cls_name, cnt in line_counts[line_idx].items():
                text = f"  {cls_name}: {cnt}"
                cv2.putText(frame, text, (W - 200, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                start_y += 18
            start_y += 10

        # saves image file
        #cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output, fourcc, 30,
                (frame.shape[1], frame.shape[0]), True)

            # some information on processing single frame
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(
                    elap * total))

        # write the output frame to disk
        writer.write(frame)

        # increase frame index
        frameIndex += 1
        
    # release the file pointers
    #print("[INFO] cleaning up...")
    if writer is not None:
        writer.release()
    vs.release()

    # Calculate performance metrics
    total_elapsed_time = time.time() - processing_start_time
    avg_frame_time = np.mean(frame_times) if frame_times else 0
    avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    min_frame_time = np.min(frame_times) if frame_times else 0
    max_frame_time = np.max(frame_times) if frame_times else 0

    # Print and save counts
    #print("\n" + "="*60)
    #print("RESULTADOS DEL CONTEO")
    #print("="*60)
    #print(f"\nTotal general: {counter}")

    # Performance report
    #print("\n" + "="*60)
    #print("RENDIMIENTO DEL SISTEMA")
    #print("="*60)
    #print(f"Dispositivo utilizado: {DEVICE.upper()}")
    #print(f"Frames procesados: {frameIndex}")
    #print(f"Tiempo total: {total_elapsed_time:.2f} segundos ({total_elapsed_time/60:.2f} minutos)")
    #print(f"FPS promedio: {avg_fps:.2f}")
    #print(f"Tiempo por frame:")
    #print(f"  - Promedio: {avg_frame_time*1000:.2f} ms")
    #print(f"  - Mínimo: {min_frame_time*1000:.2f} ms")
    #print(f"  - Máximo: {max_frame_time*1000:.2f} ms")

    if DEVICE == 'cuda':
        gpu_util = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.max_memory_allocated() > 0 else 0
        #print(f"\nUso de GPU:")
        #print(f"  - Memoria asignada: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        #print(f"  - Memoria máxima usada: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        #print(f"  - Eficiencia: {gpu_util:.1f}%")

    #print("\nTotales por clase:")
    for cls_name, cnt in class_counts.items():
        print(f"  {cls_name}: {cnt}")

    #print("\nTotales por línea:")
    for line_idx in range(len(counting_lines)):
        line_total = sum(line_counts[line_idx].values()) if line_counts[line_idx] else 0
        #print(f"\n  Línea {line_idx+1}: {line_total} total")
        for cls_name, cnt in line_counts[line_idx].items():
            #print(f"    {cls_name}: {cnt}")
            pass

    # Save counts to output/counts.txt
    try:
        os.makedirs(salida_path, exist_ok=True)
        with open(f'{salida_path}{nombre}_counts.txt', 'w', encoding='utf-8') as f:
            f.write(f"TOTAL_GENERAL:{counter}\n\n")
            
            f.write("TOTALES_POR_CLASE:\n")
            for cls_name, cnt in class_counts.items():
                f.write(f"{cls_name}:{cnt}\n")
            
            f.write("\nTOTALES_POR_LINEA:\n")
            for line_idx in range(len(counting_lines)):
                line_total = sum(line_counts[line_idx].values()) if line_counts[line_idx] else 0
                f.write(f"\nLinea_{line_idx+1}_total:{line_total}\n")
                for cls_name, cnt in line_counts[line_idx].items():
                    f.write(f"Linea_{line_idx+1}_{cls_name}:{cnt}\n")
        
        print(f"\n[INFO] Resultados guardados en {salida_path}{nombre}_counts.txt")
        print("="*60 + "\n")
    except Exception as e:
        print(f"[WARN] Could not save counts: {e}")
    return {"total": counter, "per_class": class_counts, "per_line": line_counts}

def analizar_con_lineas_predefinidas(video_path, nombre, counting_lines):
    """
    Analiza un video usando líneas de conteo predefinidas (sin interfaz interactiva)
    
    Args:
        video_path: Ruta al archivo de video
        nombre: Nombre base para archivos de salida
        counting_lines: Lista de líneas predefinidas [(x1,y1), (x2,y2)]
    
    Returns:
        Dict con resultados del análisis
    """
    global vs, writer, W, H, frameIndex, counter, line_counts, class_counts, counted_ids_per_line, memory

    # Set input and output paths
    output = salida_path + nombre + "_analizado.mp4"
    
    # Move model to GPU if available
    model.to(DEVICE)

    # Enable optimizations for GPU
    if DEVICE == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Get selected classes for detection
    selected_classes = filter_classes_by_category(classes)

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

    # initialize the video stream
    vs = cv2.VideoCapture(video_path)
    if not vs.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return {"error": "Could not open video"}

    writer = None
    (W, H) = (None, None)
    frameIndex = 0

    # try to determine the total number of frames
    try:
        total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = -1
        else:
            print(f"[INFO] {total} total frames in video")
    except:
        total = -1

    # Initialize per-line tracking structures
    line_counts = {}
    counted_ids_per_line = {}
    for i in range(len(counting_lines)):
        line_counts[i] = {}
        counted_ids_per_line[i] = set()

    # Reset counters for new video
    counter = 0
    class_counts = {}
    memory = {}

    # Performance tracking
    total_processing_time = 0
    frame_times = []
    processing_start_time = time.time()

    # loop over frames from the video file stream
    while True:
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # Run YOLOv11 inference with tracking
        start = time.time()
        results = model.track(frame, conf=confidence, iou=threshold, 
                            tracker="botsort.yaml", verbose=False, classes=selected_classes,
                            persist=True, device=DEVICE, half=(DEVICE == 'cuda'))
        end = time.time()
        
        frame_time = end - start
        frame_times.append(frame_time)
        total_processing_time += frame_time

        # Process tracking results
        boxes = []
        indexIDs = []
        classIDs = []
        previous = memory.copy()
        memory = {}

        if results[0].boxes is not None and results[0].boxes.id is not None:
            for i, (box, track_id, class_id) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.cls)):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                track_id = int(track_id.cpu().numpy())
                class_id = int(class_id.cpu().numpy())
                
                boxes.append([x1, y1, x2, y2])
                indexIDs.append(track_id)
                classIDs.append(class_id)
                memory[track_id] = [x1, y1, x2, y2]

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                (x1, y1) = (int(box[0]), int(box[1]))
                (x2, y2) = (int(box[2]), int(box[3]))

                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (px1, py1) = (int(previous_box[0]), int(previous_box[1]))
                    (px2, py2) = (int(previous_box[2]), int(previous_box[3]))
                    
                    p0 = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    p1 = (int((px1 + px2) / 2), int((py1 + py2) / 2))
                    cv2.line(frame, p0, p1, color, 3)

                    # Check intersection with each counting line
                    for line_idx, counting_line in enumerate(counting_lines):
                        if intersect(p0, p1, counting_line[0], counting_line[1]):
                            if indexIDs[i] not in counted_ids_per_line[line_idx]:
                                counter += 1
                                counted_ids_per_line[line_idx].add(indexIDs[i])
                                
                                cls_name = get_class_name(classIDs[i])
                                if cls_name not in line_counts[line_idx]:
                                    line_counts[line_idx][cls_name] = 0
                                line_counts[line_idx][cls_name] += 1
                                
                                if cls_name not in class_counts:
                                    class_counts[cls_name] = 0
                                class_counts[cls_name] += 1

                class_name = get_class_name(classIDs[i])
                text = f"ID:{indexIDs[i]} {class_name}"
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1

        # Draw all counting lines
        for line_idx, counting_line in enumerate(counting_lines):
            color = line_colors[line_idx % len(line_colors)]
            cv2.line(frame, counting_line[0], counting_line[1], color, 5)
            
            mid_x = (counting_line[0][0] + counting_line[1][0]) // 2
            mid_y = (counting_line[0][1] + counting_line[1][1]) // 2
            cv2.putText(frame, f"L{line_idx+1}", (mid_x, mid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw counters
        cv2.putText(frame, f"Total: {counter}", (W//2 - 100, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)

        # Draw per-class totals
        start_y = 30
        cv2.putText(frame, "TOTALES:", (10, start_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        start_y += 30
        for cls_name, cnt in class_counts.items():
            text = f"{cls_name}: {cnt}"
            cv2.putText(frame, text, (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            start_y += 25

        # Draw per-line counts
        start_y = 30
        for line_idx in range(len(counting_lines)):
            color = line_colors[line_idx % len(line_colors)]
            cv2.putText(frame, f"Linea {line_idx+1}:", (W - 200, start_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            start_y += 25
            
            line_total = sum(line_counts[line_idx].values()) if line_counts[line_idx] else 0
            cv2.putText(frame, f"  Total: {line_total}", (W - 200, start_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            start_y += 20
            
            for cls_name, cnt in line_counts[line_idx].items():
                text = f"  {cls_name}: {cnt}"
                cv2.putText(frame, text, (W - 200, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                start_y += 18
            start_y += 10

        # Initialize video writer
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output, fourcc, 30,
                (frame.shape[1], frame.shape[0]), True)

        writer.write(frame)
        frameIndex += 1

        # Optional: limit frames for testing
        if frameIndex >= 400:
            print("[INFO] Límite de frames alcanzado (testing)")
            break

    # Release resources
    if writer is not None:
        writer.release()
    vs.release()

    # Save results
    try:
        os.makedirs(salida_path, exist_ok=True)
        with open(f'{salida_path}{nombre}_counts.txt', 'w', encoding='utf-8') as f:
            f.write(f"TOTAL_GENERAL:{counter}\n\n")
            
            f.write("TOTALES_POR_CLASE:\n")
            for cls_name, cnt in class_counts.items():
                f.write(f"{cls_name}:{cnt}\n")
            
            f.write("\nTOTALES_POR_LINEA:\n")
            for line_idx in range(len(counting_lines)):
                line_total = sum(line_counts[line_idx].values()) if line_counts[line_idx] else 0
                f.write(f"\nLinea_{line_idx+1}_total:{line_total}\n")
                for cls_name, cnt in line_counts[line_idx].items():
                    f.write(f"Linea_{line_idx+1}_{cls_name}:{cnt}\n")
        
        print(f"[INFO] Resultados guardados en {salida_path}{nombre}_counts.txt")
    except Exception as e:
        print(f"[WARN] Could not save counts: {e}")
    
    return {"total": counter, "per_class": class_counts, "per_line": line_counts}