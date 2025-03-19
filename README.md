![](https://img.shields.io/badge/Creater-TCT-FFFF00) ![](https://img.shields.io/badge/development-python-006400) ![](https://img.shields.io/badge/Version-3.9.18-blue) ![](https://img.shields.io/badge/Tool-VSCode-222222)

# RealTimeObjectDetection_YOLOv8_py
RealTimeObjectDetection_YOLOv8/即時物件辨識

# 1. Package Introduce

## 1-1. OpenCV (cv2)
（Open Source Computer Vision Library）是一個開源的計算機視覺和機器學習軟件庫，旨在提供各種視覺任務的高效解決方案。它被廣泛應用於影像處理、物體檢測、影像分類、面部識別、計算機視覺等領域。

## 1-2. YOLOv8
YOLOv8 是 Ultralytics 推出的最新一代即時物件偵測 (Object Detection) 模型，延續了 YOLO (You Only Look Once) 系列的優勢，並進一步提升準確率、效能及多功能性。

### 🎯 主要功能:
YOLOv8 不僅是物件偵測模型，它還支援影像處理的多種任務，包括：
- 1. 物件偵測 (Object Detection) 📸
     
		○ 在圖片或影片中標記物件的位置 (Bounding Box) 並分類。

		○ 可應用於人流監控、智慧安防、工業自動化等場景。

- 2. 圖像分割 (Instance Segmentation) ✂️
     
		○ 除了物件偵測外，還能對物件進行像素級切割。

		○ 適用於醫學影像分析、場景理解等高精度應用。

- 3. 姿勢估計 (Pose Estimation) 🏃
     
		○ 可偵測人體關鍵點 (Keypoints)，用於運動分析、動作識別、人體行為研究。

-	4. 影像分類 (Image Classification) 🏷️

		○ 可辨識影像中物件的類別，例如：「狗」、「貓」、「車輛」等。

		○ 適合用於工業檢測、產品分類、醫療影像診斷。

- 5. 物件追蹤 (Object Tracking) 🎥
     
		○ 可在影片中追蹤移動中的目標，如行人、車輛、動物等。

		○ 應用於智慧交通、監控系統。

### 🚀 YOLOv8 的優勢:
- ✅ 更高準確率：使用新的 Anchor-Free 架構，提高偵測精度。
- ✅ 更快的推理速度：優化計算效能，在 CPU/GPU 上的執行速度更快。
- ✅ 支援 ONNX、TensorRT、OpenVINO：可部署至不同硬體加速推理。
- ✅ API 介面友善：內建 Python API (ultralytics)，易於整合與使用。
- ✅ 多功能整合：單一模型支援偵測、分割、追蹤等多種 AI 任務。

### 🎯 YOLOv8 應用領域:
- 智慧監控：人臉偵測、車牌辨識、異常行為監測
- 自動駕駛：車輛偵測、行人識別、交通標誌辨識
- 醫學影像：腫瘤偵測、細胞分類
- 工業檢測：產品瑕疵辨識、自動化品管
- 零售分析：顧客行為分析、貨架管理

### 📌 YOLOv8 各模型比較:

| **Item** | **模型** | **參數量 (M)** | **大小 (MB)** | **推理速度 (ms)** | **mAP@50-95 (%)** | **適用場景** |
|----------|--------------|-------------|-------------|-------------|-------------|-------------|
| **1** | **yolov8n.pt** (Nano)	 | 3.2M | 6 MB | 1~2ms | 37.3% | 超輕量，適合即時應用 (如 IoT、手機) |
| **2** | **yolov8s.pt** (Small) | 11.2M | 22 MB | 3~5ms | 44.9% | 平衡速度與準確率，適用於邊緣設備 |
| **3** | **yolov8m.pt** (Medium) | 25.9M | 49 MB | 5~8ms | 50.2% | 適合中等計算資源，如 GPU 伺服器 |
| **4** | **yolov8l.pt** (Large) | 43.7M | 77 MB | 8~12ms | 52.8% | 更高準確率，適用於雲端運算 |
| **5** | **yolov8x.pt** (Extra-Large) | 68.2M | 124 MB | 12~20ms | 53.9% | 最高準確率，適合高階 GPU |

💡 Nano (yolov8n.pt) 是最輕量級的模型，適用於低功耗裝置，如邊緣 AI 設備。

💡 X (yolov8x.pt) 則適合雲端 GPU 伺服器，以獲得最高的準確率。


### 📌 其他 YOLOv8 變體:
除了物件偵測 (yolov8n.pt、yolov8s.pt 等)，YOLOv8 還提供其他任務的專用模型：

| **Item** | **Name** | **Version** | **Function** |
|----------|--------------|-------------|-------------|
| **1** | **yolov8n-seg.pt** | 物件分割(Segmentation) | 汽車、人、物品輪廓分割 |
| **2** | **yolov8n-pose.pt** | 姿勢估計(Pose Estimation) | 人體關鍵點偵測 (如手腳位置) |
| **2** | **yolov8n-cls.pt** | 影像分類(Classification) | 辨識貓、狗、車等類別 |

-------

## 2. Install Command：
Please refer the command as below.

## 2-1. YOLOv8 
```bash
pip3 install ultralytics
```

## 2-2. opencv
```bash
pip install opencv-contrib-python
```
-------



------

## About Me
Thanks & Best Regards !

蔡承廷

​Senior Engineer of Semiconductor Product/Testing & ​Automation

Email: ​​kp924606@gmail.com

LinkedIn:https://www.linkedin.comin/tsai-cheng-ting/
