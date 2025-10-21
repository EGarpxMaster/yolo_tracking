"""
Traffic Counter GUI Application
Modern interface with video selection, progress tracking, and report generation
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import sys
import time
from pathlib import Path
import cv2
from PIL import Image
import torch
from datetime import datetime
import json
import numpy as np

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import configuration
from config import DETECTION_CONFIG, GUI_CONFIG
from ultralytics import YOLO
from coco_classes import filter_classes_by_category, get_class_name

# Set appearance
ctk.set_appearance_mode(GUI_CONFIG["theme"])
ctk.set_default_color_theme(GUI_CONFIG["color_scheme"])

# Global variables for line drawing
drawing_lines = []
current_line_point = None
temp_frame = None
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
            current_line_point = (x, y)
        else:
            drawing_lines.append([current_line_point, (x, y)])
            current_line_point = None
            
            temp_frame = param.copy()
            for i, line in enumerate(drawing_lines):
                color = line_colors[i % len(line_colors)]
                cv2.line(temp_frame, line[0], line[1], color, 3)
                cv2.circle(temp_frame, line[0], 5, color, -1)
                cv2.circle(temp_frame, line[1], 5, color, -1)
                mid_x = (line[0][0] + line[1][0]) // 2
                mid_y = (line[0][1] + line[1][1]) // 2
                cv2.putText(temp_frame, f"L{i+1}", (mid_x, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Setup Counting Lines", temp_frame)
    
    elif event == cv2.EVENT_MOUSEMOVE and current_line_point is not None:
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
    
    cv2.namedWindow("Setup Counting Lines")
    cv2.setMouseCallback("Setup Counting Lines", mouse_callback, first_frame)
    
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
                cv2.destroyWindow("Setup Counting Lines")
                return drawing_lines
        
        elif key == ord('u') or key == ord('U'):  # Undo
            if len(drawing_lines) > 0:
                drawing_lines.pop()
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
                y_offset = 30
                for instruction in instructions:
                    cv2.putText(temp_frame, instruction, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25
                cv2.imshow("Setup Counting Lines", temp_frame)
        
        elif key == ord('r') or key == ord('R'):  # Reset
            drawing_lines = []
            current_line_point = None
            temp_frame = first_frame.copy()
            y_offset = 30
            for instruction in instructions:
                cv2.putText(temp_frame, instruction, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            cv2.imshow("Setup Counting Lines", temp_frame)
        
        elif key == ord('q') or key == ord('Q') or key == 27:  # q or ESC
            cv2.destroyWindow("Setup Counting Lines")
            return []


def intersect(A, B, C, D):
    """Check if line segments AB and CD intersect"""
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


class VideoProcessor:
    """Handles video processing in a separate thread"""
    def __init__(self, input_video, output_folder, config):
        self.input_video = input_video
        self.output_folder = output_folder
        self.config = config
        self.is_running = False
        self.progress = 0
        self.status_message = ""
        self.start_time = None
        self.total_frames = 0
        self.processed_frames = 0
        self.counting_lines = []
        
        # Initialize model
        model_path = os.path.join(current_dir, "yolo11x.pt")
        self.model = YOLO(model_path)
        
        # Detect device
        self.device = 'cpu' if config.get('cpu_only', False) else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def get_progress_info(self):
        """Return current progress information"""
        return {
            "progress": self.progress,
            "status": self.status_message,
            "processed_frames": self.processed_frames,
            "total_frames": self.total_frames,
            "elapsed_time": time.time() - self.start_time if self.start_time else 0
        }
    
    def process(self):
        """Process video directly"""
        try:
            self.is_running = True
            self.start_time = time.time()
            
            # Get video info and setup
            cap = cv2.VideoCapture(self.input_video)
            if not cap.isOpened():
                return False, "No se pudo abrir el video"
            
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Read first frame for line setup
            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                return False, "No se pudo leer el primer frame"
            
            # Reset video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Setup counting lines (in main thread, before processing starts)
            self.status_message = "Configura las líneas de conteo..."
            
            # Prepare output
            video_name = Path(self.input_video).stem
            output_video = os.path.join(self.output_folder, f"{video_name}_processed.mp4")
            
            # Create output directory
            os.makedirs(self.output_folder, exist_ok=True)
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            # Get selected classes
            selected_classes = filter_classes_by_category(self.config.get('classes', 'vehicles'))
            
            # Initialize counters
            counter = 0
            line_counts = {}
            class_counts = {}
            counted_ids_per_line = {}
            memory = {}
            
            for i in range(len(self.counting_lines)):
                line_counts[i] = {}
                counted_ids_per_line[i] = set()
            
            # Colors for visualization
            np.random.seed(42)
            COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
            
            # Process frames
            self.processed_frames = 0
            self.status_message = "Procesando video..."
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection and tracking
                results = self.model.track(
                    frame,
                    conf=DETECTION_CONFIG.get('confidence', 0.5),
                    iou=DETECTION_CONFIG.get('iou_threshold', 0.3),
                    tracker="botsort.yaml",
                    verbose=False,
                    classes=selected_classes,
                    persist=True,
                    device=self.device
                )
                
                # Process results
                boxes = []
                indexIDs = []
                classIDs = []
                previous = memory.copy()
                memory = {}
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    for box, track_id, class_id in zip(results[0].boxes.xyxy, results[0].boxes.id, results[0].boxes.cls):
                        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                        track_id = int(track_id.cpu().numpy())
                        class_id = int(class_id.cpu().numpy())
                        
                        boxes.append([x1, y1, x2, y2])
                        indexIDs.append(track_id)
                        classIDs.append(class_id)
                        memory[track_id] = [x1, y1, x2, y2]
                
                # Draw boxes and check line crossings
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    if indexIDs[i] in previous:
                        prev_box = previous[indexIDs[i]]
                        p0 = ((x1 + x2) // 2, (y1 + y2) // 2)
                        p1 = ((prev_box[0] + prev_box[2]) // 2, (prev_box[1] + prev_box[3]) // 2)
                        cv2.line(frame, p0, p1, color, 3)
                        
                        # Check intersections
                        for line_idx, counting_line in enumerate(self.counting_lines):
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
                    
                    # Draw label
                    class_name = get_class_name(classIDs[i])
                    if self.config.get('show_labels', False):
                        text = f"ID:{indexIDs[i]} {class_name}"
                    else:
                        text = f"{indexIDs[i]} {class_name}"
                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw counting lines
                for line_idx, counting_line in enumerate(self.counting_lines):
                    color = line_colors[line_idx % len(line_colors)]
                    cv2.line(frame, counting_line[0], counting_line[1], color, 5)
                    mid_x = (counting_line[0][0] + counting_line[1][0]) // 2
                    mid_y = (counting_line[0][1] + counting_line[1][1]) // 2
                    cv2.putText(frame, f"L{line_idx+1}", (mid_x, mid_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                # Draw counters
                cv2.putText(frame, f"Total: {counter}", (width//2 - 100, 50), 
                           cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
                
                # Draw class counts
                start_y = 30
                cv2.putText(frame, "TOTALES:", (10, start_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                start_y += 30
                for cls_name, cnt in class_counts.items():
                    cv2.putText(frame, f"{cls_name}: {cnt}", (10, start_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    start_y += 25
                
                # Draw line counts
                start_y = 30
                for line_idx in range(len(self.counting_lines)):
                    color = line_colors[line_idx % len(line_colors)]
                    cv2.putText(frame, f"Linea {line_idx+1}:", (width - 200, start_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    start_y += 25
                    
                    line_total = sum(line_counts[line_idx].values()) if line_counts[line_idx] else 0
                    cv2.putText(frame, f"  Total: {line_total}", (width - 200, start_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    start_y += 20
                    
                    for cls_name, cnt in line_counts[line_idx].items():
                        cv2.putText(frame, f"  {cls_name}: {cnt}", (width - 200, start_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        start_y += 18
                    start_y += 10
                
                # Write frame
                writer.write(frame)
                
                # Update progress
                self.processed_frames += 1
                self.progress = self.processed_frames / self.total_frames if self.total_frames > 0 else 0
                self.status_message = f"Procesando frame {self.processed_frames}/{self.total_frames}"
            
            # Cleanup
            cap.release()
            writer.release()
            
            # Save counts
            counts_file = os.path.join(self.output_folder, f"{video_name}_counts.txt")
            with open(counts_file, 'w', encoding='utf-8') as f:
                f.write(f"TOTAL_GENERAL:{counter}\n\n")
                f.write("TOTALES_POR_CLASE:\n")
                for cls_name, cnt in class_counts.items():
                    f.write(f"{cls_name}:{cnt}\n")
                f.write("\nTOTALES_POR_LINEA:\n")
                for line_idx in range(len(self.counting_lines)):
                    line_total = sum(line_counts[line_idx].values()) if line_counts[line_idx] else 0
                    f.write(f"\nLinea_{line_idx+1}_total:{line_total}\n")
                    for cls_name, cnt in line_counts[line_idx].items():
                        f.write(f"Linea_{line_idx+1}_{cls_name}:{cnt}\n")
            
            # Store results for report
            self.results = {
                'total': counter,
                'class_counts': class_counts,
                'line_counts': line_counts,
                'counts_file': counts_file
            }
            
            if self.is_running:
                self.progress = 1.0
                self.status_message = "¡Procesamiento completado!"
                return True, output_video
            else:
                return False, "Procesamiento cancelado"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_message = f"Error: {str(e)}"
            return False, str(e)
        finally:
            self.is_running = False


class TrafficCounterGUI:
    """Main GUI Application"""
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title(GUI_CONFIG["title"])
        self.window.geometry(GUI_CONFIG["window_size"])
        
        # Make window resizable
        min_w, min_h = GUI_CONFIG["min_size"]
        self.window.minsize(min_w, min_h)
        
        # Variables
        self.input_video = tk.StringVar()
        self.output_folder = tk.StringVar(value="output")
        self.selected_classes = tk.StringVar(value="vehicles")
        self.show_labels = tk.BooleanVar(value=False)
        self.use_gpu = tk.BooleanVar(value=True)
        self.augment = tk.BooleanVar(value=False)
        
        # Processing state
        self.processor = None
        self.processing_thread = None
        self.update_job = None
        
        # Detect hardware
        self.detect_hardware()
        
        # Show welcome message
        self.show_welcome_dialog()
        
        # Create UI
        self.create_header()
        self.create_main_content()
        self.create_footer()
        
    def show_welcome_dialog(self):
        """Show welcome dialog on startup"""
        dialog = ctk.CTkToplevel(self.window)
        dialog.title("Bienvenido")
        dialog.geometry("550x500")  # Aumentado el tamaño
        dialog.transient(self.window)
        dialog.grab_set()
        dialog.resizable(True, True)  # Permitir redimensionar
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (550 // 2)
        y = (dialog.winfo_screenheight() // 2) - (500 // 2)
        dialog.geometry(f"550x500+{x}+{y}")
        
        # Create scrollable frame for content
        scrollable_frame = ctk.CTkScrollableFrame(dialog)
        scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Content container
        content_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Icon/Title
        title_label = ctk.CTkLabel(
            content_frame,
            text="🚗 Traffic Counter",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(pady=(10, 5))
        
        # Subtitle
        subtitle_label = ctk.CTkLabel(
            content_frame,
            text="YOLOv11x + BoTSORT",
            font=ctk.CTkFont(size=16)
        )
        subtitle_label.pack(pady=(0, 15))
        
        # Welcome message
        welcome_text = (
            "Bienvenido al sistema de conteo de tráfico.\n\n"
            "Esta aplicación utiliza inteligencia artificial\n"
            "para detectar y contar vehículos en videos.\n\n"
            "Características:\n"
            "• Detección multi-objeto con YOLOv11x\n"
            "• Rastreo avanzado con BoTSORT\n"
            "• Conteo por líneas personalizables\n"
        )
        
        message_label = ctk.CTkLabel(
            content_frame,
            text=welcome_text,
            font=ctk.CTkFont(size=13),
            justify="left"
        )
        message_label.pack(pady=15)
        
        # Separator
        separator = ctk.CTkFrame(content_frame, height=2, fg_color="gray30")
        separator.pack(fill="x", padx=20, pady=10)
        
        # Instructions
        instructions_label = ctk.CTkLabel(
            content_frame,
            text="Presiona 'Comenzar' para iniciar",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        instructions_label.pack(pady=5)
        
        # OK button
        ok_button = ctk.CTkButton(
            content_frame,
            text="Comenzar",
            command=dialog.destroy,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=45,
            width=220,
            fg_color="green",
            hover_color="darkgreen"
        )
        ok_button.pack(pady=15)
        
        # Wait for dialog to close
        self.window.wait_window(dialog)
        
    def detect_hardware(self):
        """Detect available hardware (GPU/CPU)"""
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.has_gpu = True
        else:
            self.gpu_name = "No GPU detectada"
            self.gpu_memory = 0
            self.has_gpu = False
            self.use_gpu.set(False)
    
    def create_header(self):
        """Create header with title and hardware info"""
        header_frame = ctk.CTkFrame(self.window, corner_radius=0, fg_color=("gray75", "gray25"))
        header_frame.pack(fill="x", padx=0, pady=0)
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="🚗 Traffic Counter",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(pady=(15, 5))
        
        # Subtitle
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="YOLOv11x + BoTSORT | Detección y Conteo de Vehículos",
            font=ctk.CTkFont(size=13)
        )
        subtitle_label.pack(pady=(0, 15))
        
    def create_main_content(self):
        """Create main content area"""
        main_frame = ctk.CTkFrame(self.window, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Left panel - Configuration
        left_panel = ctk.CTkScrollableFrame(main_frame, width=500)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.create_video_selection(left_panel)
        self.create_hardware_info(left_panel)
        self.create_detection_settings(left_panel)
        self.create_output_settings(left_panel)
        
        # Right panel - Preview and Progress
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True)
        
        self.create_preview_area(right_panel)
        self.create_progress_area(right_panel)
        
    def create_video_selection(self, parent):
        """Video input selection"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=10, pady=10)
        
        label = ctk.CTkLabel(
            frame,
            text="📹 Video de Entrada",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w"
        )
        label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # File selection
        file_frame = ctk.CTkFrame(frame, fg_color="transparent")
        file_frame.pack(fill="x", padx=15, pady=10)
        
        self.video_entry = ctk.CTkEntry(
            file_frame,
            textvariable=self.input_video,
            placeholder_text="Selecciona un archivo de video...",
            height=40,
            font=ctk.CTkFont(size=12)
        )
        self.video_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        browse_btn = ctk.CTkButton(
            file_frame,
            text="Explorar",
            command=self.browse_video,
            width=120,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        browse_btn.pack(side="right")
        
        # Video info
        self.video_info_label = ctk.CTkLabel(
            frame,
            text="📊 No se ha seleccionado ningún video",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            anchor="w"
        )
        self.video_info_label.pack(anchor="w", padx=15, pady=(5, 15))
        
    def create_hardware_info(self, parent):
        """Hardware information display"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=10, pady=10)
        
        label = ctk.CTkLabel(
            frame,
            text="⚡ Hardware Disponible",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w"
        )
        label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # GPU info
        if self.has_gpu:
            gpu_text = f"🚀 GPU Detectada: {self.gpu_name}\n💾 Memoria VRAM: {self.gpu_memory:.1f} GB"
            color = "green"
            recommendation = "✅ Se recomienda usar GPU para mejor rendimiento"
        else:
            gpu_text = "❌ No se detectó GPU compatible\n💻 Se usará CPU (procesamiento más lento)"
            color = "orange"
            recommendation = "⚠️ Instala CUDA y PyTorch GPU para mejor rendimiento"
        
        gpu_label = ctk.CTkLabel(
            frame,
            text=gpu_text,
            font=ctk.CTkFont(size=13),
            text_color=color,
            anchor="w",
            justify="left"
        )
        gpu_label.pack(anchor="w", padx=15, pady=10)
        
        recommendation_label = ctk.CTkLabel(
            frame,
            text=recommendation,
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w",
            wraplength=400
        )
        recommendation_label.pack(anchor="w", padx=15, pady=(0, 10))
        
        # GPU toggle
        if self.has_gpu:
            self.gpu_switch = ctk.CTkSwitch(
                frame,
                text="Usar Aceleración GPU",
                variable=self.use_gpu,
                font=ctk.CTkFont(size=13, weight="bold"),
                onvalue=True,
                offvalue=False
            )
            self.gpu_switch.pack(anchor="w", padx=15, pady=(5, 15))
        
    def create_detection_settings(self, parent):
        """Detection configuration"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=10, pady=10)
        
        label = ctk.CTkLabel(
            frame,
            text="🎯 Configuración de Detección",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w"
        )
        label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Classes selection
        classes_label = ctk.CTkLabel(
            frame,
            text="Categorías de objetos a detectar:",
            font=ctk.CTkFont(size=13),
            anchor="w"
        )
        classes_label.pack(anchor="w", padx=15, pady=(10, 5))
        
        classes_options = [
            ("Vehículos únicamente", "vehicles"),
            ("Personas y vehículos", "people_and_vehicles"),
            ("Todo tipo de transporte", "transportation"),
            ("Todas las clases", "all")
        ]
        
        # Create radio buttons for class selection
        self.class_radio_var = tk.StringVar(value="vehicles")
        for display_name, value in classes_options:
            radio = ctk.CTkRadioButton(
                frame,
                text=display_name,
                variable=self.class_radio_var,
                value=value,
                font=ctk.CTkFont(size=12)
            )
            radio.pack(anchor="w", padx=30, pady=5)
        
        self.selected_classes.set("vehicles")
        
        # Show labels toggle
        separator = ctk.CTkFrame(frame, height=2, fg_color="gray30")
        separator.pack(fill="x", padx=15, pady=15)
        
        self.labels_switch = ctk.CTkSwitch(
            frame,
            text="Mostrar etiquetas detalladas (ID:X clase)",
            variable=self.show_labels,
            font=ctk.CTkFont(size=12)
        )
        self.labels_switch.pack(anchor="w", padx=15, pady=5)
        
        # Augmentation toggle
        self.augment_switch = ctk.CTkSwitch(
            frame,
            text="Aumentación en tiempo de prueba (más lento, más preciso)",
            variable=self.augment,
            font=ctk.CTkFont(size=12)
        )
        self.augment_switch.pack(anchor="w", padx=15, pady=(5, 15))
        
    def create_output_settings(self, parent):
        """Output configuration"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=10, pady=10)
        
        label = ctk.CTkLabel(
            frame,
            text="💾 Carpeta de Salida",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w"
        )
        label.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Output folder
        folder_frame = ctk.CTkFrame(frame, fg_color="transparent")
        folder_frame.pack(fill="x", padx=15, pady=10)
        
        folder_entry = ctk.CTkEntry(
            folder_frame,
            textvariable=self.output_folder,
            height=40,
            font=ctk.CTkFont(size=12)
        )
        folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        folder_btn = ctk.CTkButton(
            folder_frame,
            text="Seleccionar",
            command=self.browse_output_folder,
            width=120,
            height=40,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        folder_btn.pack(side="right")
        
        # Info
        info_label = ctk.CTkLabel(
            frame,
            text="📁 Aquí se guardarán el video procesado, counts.txt y report.html",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w",
            wraplength=400
        )
        info_label.pack(anchor="w", padx=15, pady=(5, 15))
        
    def create_preview_area(self, parent):
        """Video preview and thumbnail"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="both", expand=True, padx=10, pady=(10, 5))
        
        label = ctk.CTkLabel(
            frame,
            text="🎬 Vista Previa",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        label.pack(anchor="w", padx=15, pady=(10, 5))
        
        # Preview canvas
        self.preview_label = ctk.CTkLabel(
            frame,
            text="Selecciona un video para ver la vista previa",
            width=500,
            height=300,
            fg_color="gray25",
            corner_radius=10
        )
        self.preview_label.pack(padx=15, pady=(5, 15), fill="both", expand=True)
        
    def create_progress_area(self, parent):
        """Progress tracking"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=10, pady=(5, 10))
        
        label = ctk.CTkLabel(
            frame,
            text="📊 Progreso del Procesamiento",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        label.pack(anchor="w", padx=15, pady=(10, 5))
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(frame, height=25)
        self.progress_bar.pack(padx=15, pady=10, fill="x")
        self.progress_bar.set(0)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            frame,
            text="✅ Listo para procesar",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w"
        )
        self.status_label.pack(anchor="w", padx=15, pady=5)
        
        # Time info frame
        time_frame = ctk.CTkFrame(frame, fg_color="transparent")
        time_frame.pack(fill="x", padx=15, pady=(5, 15))
        
        self.time_elapsed_label = ctk.CTkLabel(
            time_frame,
            text="⏱️ Tiempo transcurrido: --:--",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w"
        )
        self.time_elapsed_label.pack(side="left", padx=(0, 20))
        
        self.time_remaining_label = ctk.CTkLabel(
            time_frame,
            text="⏳ Tiempo estimado restante: --:--",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w"
        )
        self.time_remaining_label.pack(side="left")
        
    def create_footer(self):
        """Create footer with action buttons"""
        footer_frame = ctk.CTkFrame(self.window, corner_radius=0, fg_color=("gray75", "gray25"))
        footer_frame.pack(fill="x", padx=0, pady=0)
        
        button_container = ctk.CTkFrame(footer_frame, fg_color="transparent")
        button_container.pack(fill="x", padx=15, pady=15)
        
        # Start button
        self.start_btn = ctk.CTkButton(
            button_container,
            text="▶️ Iniciar Procesamiento",
            command=self.start_processing,
            font=ctk.CTkFont(size=15, weight="bold"),
            height=50,
            fg_color="green",
            hover_color="darkgreen"
        )
        self.start_btn.pack(side="left", padx=(0, 10), fill="x", expand=True)
        
        # Stop button (initially disabled)
        self.stop_btn = ctk.CTkButton(
            button_container,
            text="⏹️ Detener",
            command=self.stop_processing,
            font=ctk.CTkFont(size=15, weight="bold"),
            height=50,
            fg_color="red",
            hover_color="darkred",
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=10, fill="x", expand=True)
        
        # View Results button (initially disabled)
        self.results_btn = ctk.CTkButton(
            button_container,
            text="📊 Ver Resultados",
            command=self.view_results,
            font=ctk.CTkFont(size=15, weight="bold"),
            height=50,
            fg_color=("blue", "darkblue"),
            state="disabled"
        )
        self.results_btn.pack(side="left", padx=(10, 0), fill="x", expand=True)
        
    def browse_video(self):
        """Open file dialog to select video"""
        filename = filedialog.askopenfilename(
            title="Seleccionar Video",
            filetypes=[
                ("Archivos de video", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if filename:
            self.input_video.set(filename)
            self.load_video_info(filename)
            self.load_video_preview(filename)
    
    def browse_output_folder(self):
        """Select output folder"""
        folder = filedialog.askdirectory(title="Seleccionar Carpeta de Salida")
        if folder:
            self.output_folder.set(folder)
    
    def load_video_info(self, video_path):
        """Load and display video information"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            
            info_text = (f"📐 Resolución: {width}x{height} | "
                        f"⏱️ FPS: {fps} | "
                        f"⏳ Duración: {minutes}:{seconds:02d} | "
                        f"🎞️ Frames: {frame_count}")
            
            self.video_info_label.configure(text=info_text, text_color="white")
            
            cap.release()
        except Exception as e:
            self.video_info_label.configure(
                text=f"❌ Error al cargar video: {str(e)}", 
                text_color="red"
            )
    
    def load_video_preview(self, video_path):
        """Load and display first frame as preview"""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Resize frame to fit preview
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]
                
                # Calculate aspect ratio
                max_width = 500
                max_height = 300
                scale = min(max_width/w, max_height/h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                
                # Convert to CTkImage
                img = Image.fromarray(frame_resized)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))
                
                self.preview_label.configure(image=ctk_img, text="")
                self.preview_label.image = ctk_img
        except Exception as e:
            self.preview_label.configure(text=f"❌ Error en vista previa: {str(e)}")
    
    def start_processing(self):
        """Start video processing in background thread"""
        # Validate inputs
        if not self.input_video.get():
            messagebox.showerror("Error", "Por favor selecciona un archivo de video")
            return
        
        if not os.path.exists(self.input_video.get()):
            messagebox.showerror("Error", "El archivo de video no existe")
            return
        
        # Read first frame to setup counting lines
        cap = cv2.VideoCapture(self.input_video.get())
        if not cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir el video")
            return
        
        ret, first_frame = cap.read()
        cap.release()
        
        if not ret:
            messagebox.showerror("Error", "No se pudo leer el primer frame del video")
            return
        
        # Setup counting lines (interactive)
        counting_lines = setup_counting_lines(first_frame)
        
        if not counting_lines or len(counting_lines) == 0:
            messagebox.showwarning("Cancelado", "No se configuraron líneas de conteo. Proceso cancelado.")
            return
        
        # Create output folder if needed
        os.makedirs(self.output_folder.get(), exist_ok=True)
        
        # Prepare configuration
        config = {
            "classes": self.class_radio_var.get(),
            "show_labels": self.show_labels.get(),
            "cpu_only": not self.use_gpu.get(),
            "augment": self.augment.get()
        }
        
        # Disable start button, enable stop button
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.results_btn.configure(state="disabled")
        
        # Reset progress
        self.progress_bar.set(0)
        self.status_label.configure(text="🔄 Iniciando procesamiento...")
        self.time_elapsed_label.configure(text="⏱️ Tiempo transcurrido: 00:00")
        self.time_remaining_label.configure(text="⏳ Tiempo estimado restante: calculando...")
        
        # Create processor
        self.processor = VideoProcessor(
            self.input_video.get(),
            self.output_folder.get(),
            config
        )
        
        # Set counting lines
        self.processor.counting_lines = counting_lines
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.run_processing, daemon=True)
        self.processing_thread.start()
        
        # Start progress updater
        self.update_progress()
    
    def run_processing(self):
        """Run processing (called in background thread)"""
        success, result = self.processor.process()
        
        # Schedule UI update in main thread
        self.window.after(0, lambda: self.processing_finished(success, result))
    
    def update_progress(self):
        """Update progress bar and time estimates"""
        if self.processor and self.processor.is_running:
            info = self.processor.get_progress_info()
            
            # Update progress bar
            self.progress_bar.set(info["progress"])
            
            # Update status
            self.status_label.configure(text=f"🔄 {info['status']}")
            
            # Calculate and display elapsed time
            elapsed = info["elapsed_time"]
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            self.time_elapsed_label.configure(
                text=f"⏱️ Tiempo transcurrido: {elapsed_min:02d}:{elapsed_sec:02d}"
            )
            
            # Calculate and display remaining time
            if info["progress"] > 0.01:
                estimated_total = elapsed / info["progress"]
                remaining = estimated_total - elapsed
                remaining_min = int(remaining // 60)
                remaining_sec = int(remaining % 60)
                self.time_remaining_label.configure(
                    text=f"⏳ Tiempo estimado restante: {remaining_min:02d}:{remaining_sec:02d}"
                )
            
            # Schedule next update
            self.update_job = self.window.after(500, self.update_progress)
    
    def processing_finished(self, success, result):
        """Called when processing completes"""
        # Cancel update job if running
        if self.update_job:
            self.window.after_cancel(self.update_job)
            self.update_job = None
        
        if success:
            self.progress_bar.set(1.0)
            self.status_label.configure(text="✅ ¡Procesamiento completado exitosamente!")
            
            # Generate report
            self.generate_report(result)
            
            # Enable buttons
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.results_btn.configure(state="normal")
            
            messagebox.showinfo(
                "Éxito", 
                "El video ha sido procesado exitosamente.\n\n"
                "Haz clic en 'Ver Resultados' para abrir la carpeta de salida y el reporte."
            )
        else:
            self.status_label.configure(text=f"❌ Error: {result}")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            
            if result != "Cancelled by user":
                messagebox.showerror("Error", f"Error en el procesamiento:\n{result}")
    
    def stop_processing(self):
        """Stop current processing"""
        if self.processor:
            self.processor.is_running = False
            self.status_label.configure(text="⏹️ Deteniendo procesamiento...")
            self.stop_btn.configure(state="disabled")
    
    def generate_report(self, output_video_path):
        """Generate HTML report with statistics and visualizations"""
        try:
            # Read counts.txt
            counts_file = os.path.join(self.output_folder.get(), "counts.txt")
            if not os.path.exists(counts_file):
                print("Warning: counts.txt not found")
                return
            
            # Parse counts
            with open(counts_file, 'r', encoding='utf-8') as f:
                counts_data = f.read()
            
            # Parse data for better presentation
            lines = counts_data.split('\n')
            total_general = 0
            class_counts = {}
            line_counts = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    if key == 'TOTAL_GENERAL':
                        total_general = int(value.strip())
                    elif 'Linea_' in key and '_total' in key:
                        line_num = key.split('_')[1]
                        line_counts[f"Línea {line_num}"] = int(value.strip())
                    elif 'Linea_' not in key and key not in ['TOTALES_POR_CLASE', 'TOTALES_POR_LINEA', '']:
                        class_counts[key.strip()] = int(value.strip())
            
            # Generate HTML report
            report_path = os.path.join(self.output_folder.get(), "report.html")
            
            html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Conteo de Tráfico</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 30px 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .info-card {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            border-left: 5px solid #667eea;
        }}
        
        .info-card h3 {{
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        
        .info-card p {{
            font-size: 1.1em;
            color: #333;
            word-break: break-word;
        }}
        
        .total-section {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 40px;
        }}
        
        .total-section h2 {{
            font-size: 1.5em;
            margin-bottom: 15px;
        }}
        
        .total-section .count {{
            font-size: 4em;
            font-weight: bold;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
        }}
        
        .stat-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid #e9ecef;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .stat-box:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }}
        
        .stat-box .label {{
            color: #667eea;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 0.9em;
        }}
        
        .stat-box .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        
        .raw-data {{
            background: #2d3748;
            color: #e2e8f0;
            padding: 25px;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.6;
            overflow-x: auto;
            white-space: pre-wrap;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            
            .total-section .count {{
                font-size: 3em;
            }}
            
            .info-grid, .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚗 Reporte de Conteo de Tráfico</h1>
            <p>Análisis de Video con YOLOv11x + BoTSORT</p>
        </div>
        
        <div class="content">
            <!-- Information Cards -->
            <div class="info-grid">
                <div class="info-card">
                    <h3>📅 Fecha de Generación</h3>
                    <p>{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                </div>
                <div class="info-card">
                    <h3>📹 Video Procesado</h3>
                    <p>{Path(self.input_video.get()).name}</p>
                </div>
                <div class="info-card">
                    <h3>📁 Video de Salida</h3>
                    <p>{Path(output_video_path).name}</p>
                </div>
                <div class="info-card">
                    <h3>🎯 Categoría Detectada</h3>
                    <p>{self.class_radio_var.get()}</p>
                </div>
            </div>
            
            <!-- Total Count -->
            <div class="total-section">
                <h2>Conteo Total General</h2>
                <div class="count">{total_general}</div>
                <p style="margin-top: 15px; font-size: 1.1em;">objetos contabilizados en total</p>
            </div>
            
            <!-- Class Counts -->
            <div class="section">
                <h2>📊 Conteo por Clase de Objeto</h2>
                <div class="stats-grid">
"""
            
            for class_name, count in class_counts.items():
                html_content += f"""
                    <div class="stat-box">
                        <div class="label">{class_name.upper()}</div>
                        <div class="value">{count}</div>
                    </div>
"""
            
            html_content += """
                </div>
            </div>
            
            <!-- Line Counts -->
"""
            
            if line_counts:
                html_content += """
            <div class="section">
                <h2>📏 Conteo por Línea</h2>
                <div class="stats-grid">
"""
                for line_name, count in line_counts.items():
                    html_content += f"""
                    <div class="stat-box">
                        <div class="label">{line_name}</div>
                        <div class="value">{count}</div>
                    </div>
"""
                html_content += """
                </div>
            </div>
"""
            
            html_content += f"""
            <!-- Raw Data -->
            <div class="section">
                <h2>📄 Datos Completos</h2>
                <div class="raw-data">{counts_data}</div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generado por Traffic Counter | YOLOv11x + BoTSORT</p>
            <p>© 2025 - Sistema de Análisis de Tráfico</p>
        </div>
    </div>
</body>
</html>
"""
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"[INFO] Reporte generado: {report_path}")
                
        except Exception as e:
            print(f"[ERROR] Error generando reporte: {e}")
            import traceback
            traceback.print_exc()
    
    def view_results(self):
        """Open output folder and report"""
        output_folder = self.output_folder.get()
        
        if os.path.exists(output_folder):
            # Open report if exists
            report_path = os.path.join(output_folder, "report.html")
            if os.path.exists(report_path):
                import webbrowser
                webbrowser.open(f'file:///{os.path.abspath(report_path).replace(os.sep, "/")}')
            
            # Open folder in file explorer (Windows)
            try:
                os.startfile(output_folder)
            except AttributeError:
                # For non-Windows systems
                import subprocess
                if sys.platform == 'darwin':
                    subprocess.run(['open', output_folder])
                else:
                    subprocess.run(['xdg-open', output_folder])
    
    def run(self):
        """Start the GUI application"""
        self.window.mainloop()


def main():
    """Main entry point"""
    print("="*60)
    print("Iniciando Traffic Counter GUI...")
    print("="*60)
    
    app = TrafficCounterGUI()
    app.run()


if __name__ == "__main__":
    main()
