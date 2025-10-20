import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os

def analizar_matriz_confusion(conteos_reales, conteos_predichos, clases, output_dir='output_analysis'):
    """
    Analiza los resultados mediante matriz de confusión y genera reportes detallados
    """
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Convertir a arrays numpy
    y_real = np.array(conteos_reales)
    y_pred = np.array(conteos_predichos)
    
    # Calcular diferencias para clasificación
    diferencias = y_pred - y_real
    
    # Crear categorías basadas en el error
    def clasificar_error(diferencia):
        if diferencia <= -3:
            return 'Subestimación Severa'
        elif -3 < diferencia <= -1:
            return 'Subestimación Leve'
        elif -1 < diferencia < 1:
            return 'Exacto'
        elif 1 <= diferencia < 3:
            return 'Sobreestimación Leve'
        else:
            return 'Sobreestimación Severa'
    
    # Aplicar clasificación
    categorias_reales = ['Real'] * len(y_real)
    categorias_predichas = [clasificar_error(diff) for diff in diferencias]
    
    # Categorías únicas en orden lógico
    categorias_orden = ['Subestimación Severa', 'Subestimación Leve', 'Exacto', 'Sobreestimación Leve', 'Sobreestimación Severa']
    
    # Crear matriz de confusión
    matriz_conf = confusion_matrix(categorias_reales, categorias_predichas, labels=categorias_orden)
    
    # Métricas adicionales
    error_absoluto = np.abs(diferencias).mean()
    error_porcentual = (np.abs(diferencias) / (y_real + 1e-6) * 100).mean()
    exactitud = (diferencias == 0).mean() * 100
    
    return {
        'matriz_confusion': matriz_conf,
        'categorias': categorias_orden,
        'conteos_reales': y_real,
        'conteos_predichos': y_pred,
        'diferencias': diferencias,
        'error_absoluto': error_absoluto,
        'error_porcentual': error_porcentual,
        'exactitud': exactitud,
        'output_dir': output_dir
    }

def generar_sketch_matriz_confusion(resultados, titulo="Matriz de Confusión - Sistema de Conteo"):
    """
    Genera un sketch visual de la matriz de confusión
    """
    
    matriz = resultados['matriz_confusion']
    categorias = resultados['categorias']
    
    plt.figure(figsize=(12, 10))
    
    # Crear heatmap de la matriz de confusión
    sns.heatmap(matriz, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=categorias,
                yticklabels=['Real'],
                cbar_kws={'label': 'Frecuencia'},
                annot_kws={'size': 12, 'weight': 'bold'})
    
    plt.title(titulo, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Categorías de Predicción', fontsize=12, fontweight='bold')
    plt.ylabel('Ground Truth', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Guardar el sketch
    sketch_path = os.path.join(resultados['output_dir'], 'matriz_confusion_sketch.png')
    plt.savefig(sketch_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return sketch_path

def generar_reporte_detallado(resultados):
    """
    Genera un reporte detallado de análisis
    """
    
    reporte = f"""
    REPORTE DE ANÁLISIS - SISTEMA DE CONTEO
    =========================================
    
    MÉTRICAS PRINCIPALES:
    - Exactitud: {resultados['exactitud']:.2f}%
    - Error Absoluto Medio: {resultados['error_absoluto']:.2f} vehículos
    - Error Porcentual Medio: {resultados['error_porcentual']:.2f}%
    
    DISTRIBUCIÓN DE ERRORES:
    """
    
    # Análisis por categoría de error
    diferencias = resultados['diferencias']
    categorias_predichas = []
    
    for diff in diferencias:
        if diff <= -3:
            categorias_predichas.append('Subestimación Severa')
        elif -3 < diff <= -1:
            categorias_predichas.append('Subestimación Leve')
        elif -1 < diff < 1:
            categorias_predichas.append('Exacto')
        elif 1 <= diff < 3:
            categorias_predichas.append('Sobreestimación Leve')
        else:
            categorias_predichas.append('Sobreestimación Severa')
    
    distribucion = pd.Series(categorias_predichas).value_counts()
    
    for categoria, count in distribucion.items():
        porcentaje = (count / len(diferencias)) * 100
        reporte += f"    - {categoria}: {count} casos ({porcentaje:.1f}%)\n"
    
    # Análisis caso por caso
    reporte += "\nANÁLISIS CASO POR CASO:\n"
    for i, (real, pred, diff) in enumerate(zip(resultados['conteos_reales'], 
                                              resultados['conteos_predichos'], 
                                              resultados['diferencias'])):
        estado = "✓ EXACTO" if diff == 0 else f"✗ ERROR: {diff:+d}"
        reporte += f"    Caso {i+1}: Real={real}, Predicho={pred} → {estado}\n"
    
    return reporte

def guardar_resultados_completos(resultados):
    """
    Guarda todos los resultados en archivos
    """
    
    output_dir = resultados['output_dir']
    
    # Guardar reporte detallado
    reporte_path = os.path.join(output_dir, 'reporte_analisis.txt')
    with open(reporte_path, 'w', encoding='utf-8') as f:
        f.write(generar_reporte_detallado(resultados))
    
    # Guardar datos numéricos
    datos_path = os.path.join(output_dir, 'datos_numericos.csv')
    df_datos = pd.DataFrame({
        'Conteo_Real': resultados['conteos_reales'],
        'Conteo_Predicho': resultados['conteos_predichos'],
        'Diferencia': resultados['diferencias'],
        'Error_Absoluto': np.abs(resultados['diferencias'])
    })
    df_datos.to_csv(datos_path, index=False, encoding='utf-8')
    
    # Guardar matriz de confusión numérica
    matriz_path = os.path.join(output_dir, 'matriz_confusion_numerica.csv')
    df_matriz = pd.DataFrame(
        resultados['matriz_confusion'],
        index=['Real'],
        columns=resultados['categorias']
    )
    df_matriz.to_csv(matriz_path, encoding='utf-8')
    
    return reporte_path, datos_path, matriz_path

# EJEMPLO DE USO CON TUS DATOS
if __name__ == "__main__":
    # Datos de ejemplo (reemplaza con tus datos reales)
    # Estos son los conteos manuales (ground truth) y los del modelo
    
    # Ejemplo 1: Datos simples
    conteos_manuales = [17, 71, 70]
    conteos_modelo = [17, 57, 65]
    clases = ["Video 1", "Video_2", "Video_3"]
    
    # Ejemplo 2: Si tienes datos por clase de vehículo
    # conteos_manuales = [10, 25, 15, 8]   # car, bus, truck, motorcycle
    # conteos_modelo = [12, 23, 14, 7]     # car, bus, truck, motorcycle
    # clases = ["Car", "Bus", "Truck", "Motorcycle"]
    
    print(" INICIANDO ANÁLISIS POR MATRIZ DE CONFUSIÓN")
    print("=" * 50)
    
    # Realizar análisis
    resultados = analizar_matriz_confusion(
        conteos_reales=conteos_manuales,
        conteos_predichos=conteos_modelo,
        clases=clases,
        output_dir='analisis_conteo'
    )
    
    # Generar sketch de matriz de confusión
    print(" Generando matriz de confusión...")
    sketch_path = generar_sketch_matriz_confusion(
        resultados, 
        titulo="Matriz de Confusión - Sistema de Conteo YOLOv11"
    )
    
    # Guardar resultados completos
    print(" Guardando reportes...")
    reporte_path, datos_path, matriz_path = guardar_resultados_completos(resultados)
    
    # Mostrar resumen
    print("\n" + "=" * 50)
    print("ANÁLISIS COMPLETADO")
    print("=" * 50)
    print(f"Métricas Principales:")
    print(f"   • Exactitud: {resultados['exactitud']:.2f}%")
    print(f"   • Error Absoluto: {resultados['error_absoluto']:.2f} vehículos")
    print(f"   • Error Porcentual: {resultados['error_porcentual']:.2f}%")
    
    print(f"\n📁 Archivos generados:")
    print(f"   • Sketch matriz: {sketch_path}")
    print(f"   • Reporte completo: {reporte_path}")
    print(f"   • Datos numéricos: {datos_path}")
    print(f"   • Matriz numérica: {matriz_path}")
    
    # Mostrar matriz de confusión en consola
    print(f"\n📋 Matriz de Confusión:")
    df_matriz = pd.DataFrame(
        resultados['matriz_confusion'],
        index=['Real'],
        columns=resultados['categorias']
    )
    print(df_matriz)
    
    # Análisis detallado de errores
    print(f"\n🔍 DISTRIBUCIÓN DE ERRORES:")
    diferencias = resultados['diferencias']
    for i, (real, pred, diff) in enumerate(zip(resultados['conteos_reales'], 
                                              resultados['conteos_predichos'], 
                                              diferencias)):
        if diff == 0:
            print(f"   ✓ Frame {i+1}: EXACTO (Real={real}, Predicho={pred})")
        elif diff < 0:
            print(f"   ▼ Frame {i+1}: SUBESTIMACIÓN {abs(diff)} (Real={real}, Predicho={pred})")
        else:
            print(f"   ▲ Frame {i+1}: SOBREESTIMACIÓN {diff} (Real={real}, Predicho={pred})")