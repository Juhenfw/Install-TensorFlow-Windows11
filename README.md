# Install TensorFlow di Windows 11
Tutorial instalasi TensorFlow pada lingkungan Windows 10 atau di atasnya. Pada Repository ini juga akan dijelaskan terkait konversi dari model PyTorch (.pt) ke model TensorFlow (.onxx & .engine)

## Spesifikasi yang saya Gunakan
- Windows 11
- NVIDIA GPU RTX 4060 dengan versi driver terbaru
- CUDA versi 12.1
- Python dengan versi 3.10
- TensorRT 10

## Langkah Awal
1. Unduh NVIDIA TensorRT pada [website resmi](https://developer.nvidia.com/tensorrt/download)
2. Pada Bagian Available Version, pilih versi TensorRT yang dibutuhkan (misal: TensorRT 10)
3. Klik persetujuan lalu pilih TensorRT versi terbarunya
4. Pada bagian Zip Packages for Windows klik "TensorRT 10.x GA for Windows 10, 11, Server 2022 and CUDA 12.0 to 12.9 ZIP Package" (karena saya menggunakan CUDA versi 12.1)
5. Tunggu hingga unduhan selesai

## Langkah Kedua
1. Buka File Explorer
2. Masuk ke lokasi unduhan file TensorRT, lalu extract file
3. Setelah extract, akan muncul folder `TensorRT-10.x.x.x`
4. Buka folder tersebut dan masuk ke folder `lib`
5. **Copy semua file DLL** yang ada di dalam folder `lib`

## Langkah Ketiga - Copy DLL lalu Paste ke CUDA Directory
1. Buka File Explorer dan navigasi ke direktori instalasi CUDA:
```cmd
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
```
3. **Paste semua file DLL** yang sudah di-copy dari TensorRT ke folder `bin` CUDA
4. Ini adalah metode yang lebih mudah daripada menambahkan PATH environment variable

## Langkah Keempat - Install TensorRT Python Package
1. Buka Command Prompt atau PowerShell
2. Masuk ke dalam virtual environment, jika belum ada
```cmd
python -m venv pytorch-tensorflow-env
```
3. Aktifkan virtual environment, contoh:
```cmd
pytorch-tensorflow-env\Scripts\activate
```
4. Navigasi ke folder TensorRT yang sudah di-extract
5. Masuk ke folder `python`
6. Install TensorRT wheel file sesuai versi Python Anda:
```cmd
python.exe -m pip install tensorrt-*-cp310-none-win_amd64.whl
```

(ganti `cp310` dengan versi Python Anda, misal `cp310` untuk Python 3.10)

(ganti `*` dengan versi TensorRT, misal 10.10.0.31)

contoh lengkapnya:
```cmd
python.exe -m pip install tensorrt-10.10.0.31-cp310-none-win_amd64.whl
```

7. **Opsional**: Install TensorRT lean dan dispatch runtime:
```cmd
python.exe -m pip install tensorrt_lean--cp310-none-win_amd64.whl
python.exe -m pip install tensorrt_dispatch--cp310-none-win_amd64.whl
```

## Langkah Kelima - Install TensorFlow
1. Install TensorFlow dengan GPU support:
```cmd
pip install tensorflow==2.10.0
```
2. Atau install tensorflow-gpu secara eksplisit
```cmd
pip install tensorflow-gpu==2.10.0
```


## Verifikasi Instalasi

### Test TensorRT:
```python
import tensorrt as trt
print("TensorRT version:", trt.version)

### Test TensorFlow:
import tensorflow as tf
print("TensorFlow version:", tf.version)
print("GPU devices:", tf.config.list_physical_devices('GPU'))


### Test lengkap GPU Support:
import tensorflow as tf
import tensorrt as trt

print("=" * 50)
print("SYSTEM CHECK")
print("=" * 50)
print("TensorFlow version:", tf.version)
print("TensorRT version:", trt.version)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
print("CUDA built with TensorFlow:", tf.test.is_built_with_cuda())

Test simple computation
if tf.config.list_physical_devices('GPU'):
with tf.device('/GPU:0'):
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
c = tf.matmul(a, b)
print("GPU computation result:")
print(c)
```

## Konversi Model PyTorch ke TensorFlow

### Langkah 1: Export PyTorch ke ONNX
```python
import torch
import torch.onnx

Load PyTorch model
model = torch.load('model.pt')
model.eval()

Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

Export to ONNX
torch.onnx.export(
model,
dummy_input,
"model.onnx",
verbose=True,
input_names=['input'],
output_names=['output']
)
```

### Langkah 2: Convert ONNX ke TensorRT Engine
```python
import tensorrt as trt
import numpy as np

def build_engine(onnx_file_path, engine_file_path):
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# Parse ONNX file
with open(onnx_file_path, 'rb') as model:
    if not parser.parse(model.read()):
        print('ERROR: Failed to parse the ONNX file.')
        return None

# Build engine
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

serialized_engine = builder.build_serialized_network(network, config)

# Save engine to file
with open(engine_file_path, 'wb') as f:
    f.write(serialized_engine)

return serialized_engine
Convert model
build_engine('model.onnx', 'model.engine')
```

## Troubleshooting

### Error "DLL load failed":
- Pastikan semua DLL TensorRT sudah di-copy ke folder CUDA bin[1]
- Restart Command Prompt/PowerShell setelah instalasi

### Error "No module named 'tensorrt'":
- Pastikan wheel file sudah ter-install dengan benar
- Cek apakah versi Python sesuai dengan wheel file

### GPU tidak terdeteksi:
- Verifikasi driver NVIDIA versi terbaru terinstall
- Pastikan CUDA versi 12.1 terinstall dengan benar
- Restart komputer setelah instalasi driver

## Dependencies yang Dibutuhkan
```cmd
tensorflow==2.15.0
tensorrt
numpy
opencv-python
pillow
```

## Struktur Project
```cmd
ai-conversion-project/
├── models/
│ ├── pytorch/
│ ├── onnx/
│ └── tensorrt/
├── scripts/
│ ├── convert_pytorch_to_onnx.py
│ ├── convert_onnx_to_tensorrt.py
│ └── inference_tensorrt.py
├── requirements.txt
└── README.md
```
## Referensi
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [TensorFlow GPU Support Guide](https://www.tensorflow.org/install/gpu)
- [PyTorch to ONNX Conversion](https://pytorch.org/docs/stable/onnx.html)

## Kontribusi
Jika Anda menemukan masalah atau ingin berkontribusi, silakan buat issue atau pull request.

## Lisensi
MIT License

---

**Happy Coding! 🚀**
