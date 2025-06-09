# **Install TensorFlow di Windows 11**
Panduan instalasi TensorFlow pada lingkungan Windows 10 atau di atasnya. Dalam repository ini juga akan dijelaskan cara mengonversi model PyTorch (.pt) ke model TensorFlow (.onnx & .engine).

## **Daftar Isi**
1. [Spesifikasi pada Perangkat yang Saya Gunakan](#spesifikasi-pada-perangkat-yang-saya-gunakan)
2. [Langkah Awal: Instalasi TensorRT](#langkah-awal-instalasi-tensorrt)
   - Unduh NVIDIA TensorRT
   - Extract File TensorRT]
   - Copy DLL File ke CUDA Directory
3. [Langkah Kedua: Install TensorRT Python Package](#langkah-kedua-install-tensorrt-python-package)
   - Membuat Virtual Environment
   - Instal TensorRT Python Package
4. [Langkah Ketiga: Install TensorFlow](#langkah-ketiga-install-tensorflow)
   - Install TensorFlow dengan GPU Support
   - Downgrade NumPy (Opsional)
5. [Verifikasi Instalasi](#verifikasi-instalasi)
   - Verifikasi TensorRT
   - Verifikasi TensorFlow
   - Verifikasi Dukungan GPU
6. [Konversi Model PyTorch ke TensorFlow](#konversi-model-pytorch-ke-tensorflow)
   - Export PyTorch ke ONNX
   - Konversi ONNX ke TensorRT Engine
   - Konversi Langsung dari PyTorch ke TensorRT Engine
7. [Troubleshooting](#troubleshooting)
8. [Dependencies yang Dibutuhkan](#dependencies-yang-dibutuhkan)
9. [Struktur Project](#struktur-project)
10. [Referensi](#referensi)
11. [Kontribusi](#kontribusi)
12. [Lisensi](#lisensi)

---

## **Spesifikasi pada Perangkat yang Saya Gunakan**
- **OS**: Windows 11
- **GPU**: NVIDIA RTX 4060 dengan versi driver terbaru
- **CUDA**: Versi 12.1
- **Python**: Versi 3.10
- **TensorRT**: Versi 10

---

## **Langkah Awal: Instalasi TensorRT**

### 1. **Unduh NVIDIA TensorRT**
   - Kunjungi website resmi TensorRT:
     https://developer.nvidia.com/tensorrt/download
     
   - Pilih versi TensorRT yang sesuai dengan kebutuhan (misal: TensorRT 10).
   - Setujui ketentuan dan pilih versi terbaru dari TensorRT.
   - Pada bagian *Zip Packages for Windows*, klik "TensorRT 10.x GA for Windows 10, 11, Server 2022 and CUDA 12.0 to 12.9 ZIP Package" (sesuaikan dengan CUDA versi yang digunakan).

### 2. **Extract File TensorRT**
   - Buka *File Explorer* dan navigasi ke lokasi unduhan TensorRT.
   - Extract file yang telah diunduh.
   - Setelah extract, akan muncul folder `TensorRT-10.x.x.x`.
   - Buka folder tersebut dan masuk ke folder `lib`.

### 3. **Copy DLL File ke CUDA Directory**
   - **Copy semua file DLL** yang ada di folder `lib` dan paste ke direktori CUDA:
     ```cmd
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
     ```
     Folder `lib`:
     
     <img src="https://github.com/Juhenfw/Install-TensorFlow-Windows11/blob/eb1f985b2469e691114e2bf0b7da88a01cc1ca94/pic/pathtensorrt.png" width="600" height="350">
     
     Direktori CUDA:
     
     <img src="https://github.com/Juhenfw/Install-TensorFlow-Windows11/blob/main/pic/pathcuda.png" width="600" height="400">

   - **Catatan**: Ini adalah metode yang lebih mudah daripada menambahkan *PATH environment variable* secara manual.

---

## **Langkah Kedua: Install TensorRT Python Package**

### 1. **Membuat Virtual Environment**
   - Buka Command Prompt atau PowerShell, lalu buat virtual environment baru:
     ```cmd
     python -m venv pytorch-tensorflow-env
     ```

   - Aktifkan virtual environment:
     ```cmd
     pytorch-tensorflow-env\Scripts\activate
     ```

### 2. **Instal TensorRT Python Package**
   - Navigasi ke folder TensorRT yang sudah di-extract, lalu masuk ke folder `python`.
     
     <img src="https://github.com/Juhenfw/Install-TensorFlow-Windows11/blob/main/pic/pythontensorrt.png" width="600" height="450">
   - Install file *wheel* TensorRT sesuai dengan versi Python Anda:
     ```cmd
     python.exe -m pip install tensorrt-*-cp310-none-win_amd64.whl
     ```
     Ganti `*` dengan versi TensorRT, misal dengan direktori lengkap letak TensorRT:
     ```cmd
     python.exe -m pip install tensorrt-10.10.0.31-cp310-none-win_amd64.whl
     ```
     ```cmd
     python.exe -m pip install C:\Users\YMPI\Downloads\TensorRT-10.10.0.31.Windows.win10.cuda-12.9\TensorRT-10.10.0.31\python\tensorrt-10.10.0.31-cp310-none-win_amd64.whl
     ```

   - **Opsional**: Install TensorRT lean dan dispatch runtime:
     ```cmd
     python.exe -m pip install tensorrt_lean--cp310-none-win_amd64.whl
     python.exe -m pip install tensorrt_dispatch--cp310-none-win_amd64.whl
     ```
     Atau
     ```cmd
     python.exe -m pip install C:\Users\YMPI\Downloads\TensorRT-10.10.0.31.Windows.win10.cuda-12.9\TensorRT-10.10.0.31\python\tensorrt_lean--cp310-none-win_amd64.whl
     python.exe -m pip install C:\Users\YMPI\Downloads\TensorRT-10.10.0.31.Windows.win10.cuda-12.9\TensorRT-10.10.0.31\python\tensorrt_dispatch--cp310-none-win_amd64.whl
     ```

---

## **Langkah Ketiga: Install TensorFlow**

### 1. **Install TensorFlow dengan GPU Support**
   - Install TensorFlow dengan dukungan GPU:
     ```cmd
     pip install tensorflow==2.10.0
     ```

### 2. **Downgrade NumPy (Opsional)**
   - Beberapa versi TensorFlow membutuhkan NumPy versi tertentu. Jika perlu, downgrade NumPy ke versi 1.x:
     ```cmd
     pip install "numpy<2.0"
     ```

---

## **Verifikasi Instalasi**

### 1. **Verifikasi TensorRT**
   ```python
   import tensorrt as trt
   print("TensorRT version:", trt.version)
   ```

### 2. **Verifikasi TensorFlow**
   ```python
   import tensorflow as tf
   print("TensorFlow version:", tf.version)
   print("GPU devices:", tf.config.list_physical_devices('GPU'))
   ```

### 3. **Verifikasi Dukungan GPU**
   ```python
   import tensorflow as tf
   import tensorrt as trt

   print("=" * 50)
   print("SYSTEM CHECK")
   print("=" * 50)
   print("TensorFlow version:", tf.version)
   print("TensorRT version:", trt.version)
   print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
   print("CUDA built with TensorFlow:", tf.test.is_built_with_cuda())

   **Test perhitungan sederhana pada GPU:**
   if tf.config.list_physical_devices('GPU'):
       with tf.device('/GPU:0'):
           a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
           b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
           c = tf.matmul(a, b)
           print("GPU computation result:")
           print(c)
   ```

---

## **Konversi Model PyTorch ke TensorFlow**

### **Langkah 1: Export PyTorch ke ONNX**
   ```python
   import torch
   import torch.onnx

   # Load model PyTorch
   model = torch.load('model.pt')
   model.eval()

   # Membuat dummy input
   dummy_input = torch.randn(1, 3, 224, 224)

   # Export ke ONNX
   torch.onnx.export(
       model,
       dummy_input,
       "model.onnx",
       verbose=True,
       input_names=['input'],
       output_names=['output']
   )
   ```

### **Langkah 2: Konversi ONNX ke TensorRT Engine**
   ```python
   import tensorrt as trt
   import numpy as np

   def build_engine(onnx_file_path, engine_file_path):
       logger = trt.Logger(trt.Logger.WARNING)
       builder = trt.Builder(logger)
       network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
       parser = trt.OnnxParser(network, logger)

       # Parsing file ONNX
       with open(onnx_file_path, 'rb') as model:
           if not parser.parse(model.read()):
               print('ERROR: Failed to parse the ONNX file.')
               return None

       # Membangun engine
       config = builder.create_builder_config()
       config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

       serialized_engine = builder.build_serialized_network(network, config)

       # Menyimpan engine ke file
       with open(engine_file_path, 'wb') as f:
           f.write(serialized_engine)

       return serialized_engine

   # Konversi model
   build_engine('model.onnx', 'model.engine')
   ```

### **Langkah 3: Konversi Langsung dari PyTorch ke TensorRT Engine**
   ```python
   from ultralytics import YOLO

   model = YOLO('yolo11s.pt')

   # Export dengan optimasi maksimal untuk kecepatan
   model.export(
       format='engine',
       device=0,
       half=True,           # FP16 untuk kecepatan
       batch=32,           # Ukuran batch untuk throughput
       workspace=4,        # Memori workspace (GB)
       imgsz=640,          # Ukuran input
       verbose=False
   )
   ```

---

## **Troubleshooting**

### **Error "DLL load failed":**
   - Pastikan semua file DLL TensorRT sudah dicopy ke folder `bin` CUDA.
   - Restart Command Prompt/PowerShell setelah instalasi.

### **Error "No module named 'tensorrt'":**
   - Pastikan file *wheel* TensorRT sudah ter-install dengan benar.
   - Cek apakah versi Python yang digunakan sesuai dengan file *wheel*.

### **GPU Tidak Terdeteksi:**
   - Pastikan driver NVIDIA terbaru sudah terinstal.
   - Verifikasi bahwa CUDA versi 12.1 sudah terpasang dengan benar.
   - Restart komputer setelah instalasi driver.

---

## **Dependencies yang Dibutuhkan**
   ```cmd
   tensorflow==2.10.0
   tensorrt
   numpy==1.24.3
   opencv-python
   pillow
   protobuf==3.20.3
   onnx
   onnxruntime
   ```

---

## **Struktur Project**
   ```cmd
   ai-conversion-project/
   ├── models/
   │   ├── pytorch/
   │   ├── onnx/
   │   └── tensorrt/
   ├── scripts/
   │   ├── convert_pytorch_to_onnx.py
   │   ├── convert_onnx_to_tensorrt.py
   │   └── inference_tensorrt.py
   ├── requirements.txt
   └── README.md
   ```
---

## **Referensi**
   - [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
   - [TensorFlow GPU Support Guide](https://www.tensorflow.org/install/gpu)
   - [PyTorch to ONNX Conversion](https://pytorch.org/docs/stable/onnx.html)

---

## **Kontribusi**
Jika Anda menemukan masalah atau ingin berkontribusi, silakan buat *issue* atau *pull request*.

## **Lisensi**
MIT License

---
Jika Anda ingin mengakses dan mencoba *Test Code*: [Klik Di sini](https://github.com/Juhenfw/Install-TensorFlow-Windows11/tree/main/src)

**Happy Coding! 🚀**
