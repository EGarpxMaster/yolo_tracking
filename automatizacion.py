import os 
import time 
import cv2

# Obtener el directorio actual del script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path donde se encuentran los videos de entrada (relativo al directorio del script)
videos_path = os.path.join(current_dir, "input")

def listar_videos():
    """
    Lista todos los archivos de video en el directorio de entrada
    """
    if not os.path.exists(videos_path):
        print(f"[ERROR] El directorio de entrada no existe: {videos_path}")
        print(f"[INFO] Creando directorio: {videos_path}")
        os.makedirs(videos_path, exist_ok=True)
        return []
    
    archivos_video = [f for f in os.listdir(videos_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    print(f"Videos disponibles para análisis: {archivos_video}")
    
    return archivos_video

def configurar_lineas_para_todos():
    """
    Configura las líneas de conteo para todos los videos antes de iniciar el análisis
    Retorna un diccionario con las líneas configuradas para cada video
    """
    from analizar_video import setup_counting_lines
    
    videos = listar_videos()
    configuraciones = {}
    
    print("\n" + "="*60)
    print("CONFIGURACIÓN DE LÍNEAS DE CONTEO")
    print("="*60)
    print(f"\nSe configurarán líneas para {len(videos)} video(s)")
    print("Por favor, dibuja las líneas para cada video...\n")
    
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Configurando líneas para: {video}")
        print("-" * 60)
        
        # Abrir el video y leer el primer frame
        ruta_video = os.path.join(videos_path, video)
        vs = cv2.VideoCapture(ruta_video)
        
        if not vs.isOpened():
            print(f"[ERROR] No se pudo abrir el video: {video}")
            continue
            
        ret, first_frame = vs.read()
        vs.release()
        
        if not ret:
            print(f"[ERROR] No se pudo leer el primer frame de: {video}")
            continue
        
        # Configurar líneas para este video
        lineas = setup_counting_lines(first_frame)
        configuraciones[video] = lineas
        
        print(f"[✓] {len(lineas)} línea(s) configurada(s) para {video}")
    
    print("\n" + "="*60)
    print(f"CONFIGURACIÓN COMPLETA - {len(configuraciones)} videos listos")
    print("="*60)
    print("\nIniciando análisis automático...\n")
    
    return configuraciones

# ============================================================================
# FASE 1: CONFIGURACIÓN DE LÍNEAS (requiere intervención humana)
# ============================================================================
print("\n FASE 1: CONFIGURACIÓN DE LÍNEAS DE CONTEO")
print("=" * 60)
configuraciones_lineas = configurar_lineas_para_todos()

# ============================================================================
# FASE 2: ANÁLISIS AUTOMÁTICO (sin intervención humana)
# ============================================================================
print("\n FASE 2: ANÁLISIS AUTOMÁTICO DE VIDEOS")
print("=" * 60)
time_0 = time.time()

from analizar_video import analizar_con_lineas_predefinidas

for video, lineas in configuraciones_lineas.items():
    print(f"\n📹 Analizando video: {video}")
    print(f"   Líneas configuradas: {len(lineas)}")
    
    ruta_video = os.path.join(videos_path, video)
    resultados = analizar_con_lineas_predefinidas(ruta_video, video, lineas)
    
    print(f"✓ Completado: {video}")
    print(f"  Total contado: {resultados['total']}")
    print(f"  Clases detectadas: {list(resultados['per_class'].keys())}")
    print("-" * 60)

time_1 = time.time()
tiempo_total = (time_1 - time_0)/60

print("\n" + "="*60)
print("ANÁLISIS COMPLETADO")
print("="*60)
print(f"Videos procesados: {len(configuraciones_lineas)}")
print(f"Tiempo total de análisis: {tiempo_total:.2f} minutos")
print(f"Tiempo promedio por video: {tiempo_total/len(configuraciones_lineas):.2f} minutos")
print("="*60 + "\n")