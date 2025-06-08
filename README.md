# **Install TensorFlow on Windows 11**
A guide to install TensorFlow on Windows 10 or later environments. This repository also explains how to convert PyTorch models (.pt) to TensorFlow models (.onnx & .engine).

## **Table of Contents**
1. [Specifications of the Hardware I Used](#specifications-of-the-hardware-i-used)
2. [Step 1: Installing TensorRT](#step-1-installing-tensorrt)
   - Download NVIDIA TensorRT
   - Extract TensorRT Files
   - Copy DLL Files to CUDA Directory
3. [Step 2: Install TensorRT Python Package](#step-2-install-tensorrt-python-package)
   - Create a Virtual Environment
   - Install TensorRT Python Package
4. [Step 3: Install TensorFlow](#step-3-install-tensorflow)
   - Install TensorFlow with GPU Support
   - Downgrade NumPy (Optional)
5. [Verify Installation](#verify-installation)
   - Verify TensorRT
   - Verify TensorFlow
   - Verify GPU Support
6. [Convert PyTorch Model to TensorFlow](#convert-pytorch-model-to-tensorflow)
   - Export PyTorch to ONNX
   - Convert ONNX to TensorRT Engine
   - Directly Convert from PyTorch to TensorRT Engine
7. [Troubleshooting](#troubleshooting)
8. [Required Dependencies](#required-dependencies)
9. [Project Structure](#project-structure)
10. [References](#references)
11. [Contributions](#contributions)
12. [License](#license)

---

## **Specifications of the Hardware I Used**
- **OS**: Windows 11
- **GPU**: NVIDIA RTX 4060 with the latest driver
- **CUDA**: Version 12.1
- **Python**: Version 3.10
- **TensorRT**: Version 10

---

## **Step 1: Installing TensorRT**

### 1. **Download NVIDIA TensorRT**
   - Visit the official TensorRT website:
     https://developer.nvidia.com/tensorrt/download
     
   - Choose the appropriate version of TensorRT (e.g., TensorRT 10).
   - Agree to the terms and select the latest version of TensorRT.
   - Under *Zip Packages for Windows*, click "TensorRT 10.x GA for Windows 10, 11, Server 2022 and CUDA 12.0 to 12.9 ZIP Package" (adjust according to your CUDA version).

### 2. **Extract TensorRT Files**
   - Open *File Explorer* and navigate to the TensorRT download location.
   - Extract the downloaded file.
   - After extraction, a folder `TensorRT-10.x.x.x` will appear.
   - Open that folder and go to the `lib` folder.

### 3. **Copy DLL Files to CUDA Directory**
   - **Copy all the DLL files** from the `lib` folder and paste them into the CUDA directory:
     ```cmd
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
     ```
     `lib` folder:
     
     <img src="https://github.com/Juhenfw/Install-TensorFlow-Windows11/blob/eb1f985b2469e691114e2bf0b7da88a01cc1ca94/pic/pathtensorrt.png" width="600" height="350">
     
     CUDA directory:
     
     <img src="https://github.com/Juhenfw/Install-TensorFlow-Windows11/blob/main/pic/pathcuda.png" width="600" height="400">

   - **Note**: This method is easier than manually adding the *PATH environment variable*.

---

## **Step 2: Install TensorRT Python Package**

### 1. **Create a Virtual Environment**
   - Open Command Prompt or PowerShell, and create a new virtual environment:
     ```cmd
     python -m venv pytorch-tensorflow-env
     ```

   - Activate the virtual environment:
     ```cmd
     pytorch-tensorflow-env\Scripts\activate
     ```

### 2. **Install TensorRT Python Package**
   - Navigate to the extracted TensorRT folder, then enter the `python` folder.
     
     <img src="https://github.com/Juhenfw/Install-TensorFlow-Windows11/blob/main/pic/pythontensorrt.png" width="600" height="450">
   - Install the TensorRT wheel file according to your Python version:
     ```cmd
     python.exe -m pip install tensorrt-*-cp310-none-win_amd64.whl
     ```
     Replace `*` with the TensorRT version, for example:
     ```cmd
     python.exe -m pip install tensorrt-10.10.0.31-cp310-none-win_amd64.whl
     ```

   - **Optional**: Install TensorRT lean and dispatch runtime:
     ```cmd
     python.exe -m pip install tensorrt_lean--cp310-none-win_amd64.whl
     python.exe -m pip install tensorrt_dispatch--cp310-none-win_amd64.whl
     ```

---

## **Step 3: Install TensorFlow**

### 1. **Install TensorFlow with GPU Support**
   - Install TensorFlow with GPU support:
     ```cmd
     pip install tensorflow==2.10.0
     ```

### 2. **Downgrade NumPy (Optional)**
   - Some versions of TensorFlow require a specific version of NumPy. If needed, downgrade NumPy to version 1.x:
     ```cmd
     pip install "numpy<2.0"
     ```

---

## **Verify Installation**

### 1. **Verify TensorRT**
   ```python
   import tensorrt as trt
   print("TensorRT version:", trt.version)
   ```

### 2. **Verify TensorFlow**
   ```python
   import tensorflow as tf
   print("TensorFlow version:", tf.version)
   print("GPU devices:", tf.config.list_physical_devices('GPU'))

   ```

### 3. **Verify GPU Support**
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
   
   **Test simple computation on GPU:**
   if tf.config.list_physical_devices('GPU'):
       with tf.device('/GPU:0'):
           a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
           b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
           c = tf.matmul(a, b)
           print("GPU computation result:")
           print(c)

   ```

---

## **Convert PyTorch Model to TensorFlow**

### **Step 1: Export PyTorch to ONNX**
   ```python
   import torch
   import torch.onnx
   
   # Load PyTorch model
   model = torch.load('model.pt')
   model.eval()
   
   # Create dummy input
   dummy_input = torch.randn(1, 3, 224, 224)
   
   # Export to ONNX
   torch.onnx.export(
       model,
       dummy_input,
       "model.onnx",
       verbose=True,
       input_names=['input'],
       output_names=['output']
   )
   ```

### **Step 2: Convert ONNX to TensorRT Engine**
   ```python
   import tensorrt as trt
   import numpy as np
   
   def build_engine(onnx_file_path, engine_file_path):
       logger = trt.Logger(trt.Logger.WARNING)
       builder = trt.Builder(logger)
       network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
       parser = trt.OnnxParser(network, logger)
   
       # Parsing ONNX file
       with open(onnx_file_path, 'rb') as model:
           if not parser.parse(model.read()):
               print('ERROR: Failed to parse the ONNX file.')
               return None
   
       # Building engine
       config = builder.create_builder_config()
       config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
   
       serialized_engine = builder.build_serialized_network(network, config)
   
       # Saving engine to file
       with open(engine_file_path, 'wb') as f:
           f.write(serialized_engine)
   
       return serialized_engine
   
   # Convert model
   build_engine('model.onnx', 'model.engine')
   ```

### **Step 3: Convert Directly from PyTorch to TensorRT Engine**
   ```python
   from ultralytics import YOLO

   model = YOLO('yolo11s.pt')
   
   # Export with maximum optimization for speed
   model.export(
       format='engine',
       device=0,
       half=True,          # FP16 for speed
       batch=32,           # Adjust Batch size for throughput
       workspace=4,        # Workspace memory (GB)
       imgsz=640,          # Input size
       verbose=False
   )

   ```

---

## **Troubleshooting**

### **Error "DLL load failed":**
   - Make sure all TensorRT DLL files are copied to the `bin` folder of CUDA.
   - Restart Command Prompt/PowerShell setelah instalasi.

### **Error "No module named 'tensorrt'":**
   - Ensure the TensorRT wheel file is installed correctly.
   - Check if your Python version matches the wheel file version.

### **GPU Not Detected:**
   - Ensure the latest NVIDIA driver is installed.
   - Verify that CUDA version 12.1 is installed correctly.
   - Restart the computer after installing the driver.

---

## **Required Dependencies**
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

## **References**
   - [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
   - [TensorFlow GPU Support Guide](https://www.tensorflow.org/install/gpu)
   - [PyTorch to ONNX Conversion](https://pytorch.org/docs/stable/onnx.html)

---

## **Contributions**
If you encounter any issues or wish to contribute, please create an *issue* or pull *request*.

## **License**
MIT License

---

**Happy Coding! 🚀**
