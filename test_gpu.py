"""
Script de prueba para verificar configuración de GPU
"""
import torch

print("="*60)
print("TEST DE CONFIGURACIÓN GPU")
print("="*60)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA disponible: {cuda_available}")

if cuda_available:
    print(f"✓ GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ Número de GPUs: {torch.cuda.device_count()}")
    
    # Memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✓ Memoria total GPU: {total_memory:.2f} GB")
    
    # Test tensor on GPU
    print("\n⚡ Probando operación en GPU...")
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = x @ y  # Matrix multiplication on GPU
        print("✓ Operación en GPU exitosa!")
        
        # Clean up
        del x, y, z
        torch.cuda.empty_cache()
        
        print("\n" + "="*60)
        print("✅ CONFIGURACIÓN GPU CORRECTA")
        print("El script está listo para usar aceleración GPU")
        print("="*60)
    except Exception as e:
        print(f"✗ Error en operación GPU: {e}")
else:
    print("\n⚠ GPU no disponible")
    print("\nPara habilitar GPU:")
    print("1. Verifica que tienes una GPU NVIDIA con `nvidia-smi`")
    print("2. Instala PyTorch con CUDA:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("\n" + "="*60)
    print("ℹ El script funcionará en CPU (más lento)")
    print("="*60)