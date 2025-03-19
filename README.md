![](https://img.shields.io/badge/Creater-TCT-FFFF00) ![](https://img.shields.io/badge/development-python-006400) ![](https://img.shields.io/badge/Version-3.9.18-blue) ![](https://img.shields.io/badge/Tool-VSCode-222222)

# RealTimeObjectDetection_YOLOv8_py
RealTimeObjectDetection_YOLOv8/即時物件辨識

![image](https://github.com/user-attachments/assets/15616763-6b85-4dec-9635-2deed89f8b6d)

![image](https://github.com/user-attachments/assets/ed724ee4-3fd4-48ca-9c8a-3bccb81901cd)


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

## 3. py Code：

## 3-1. 01_ObjectDetection.py

使用 YOLOv8 進行物體檢測，並將偵測到的物體進行模糊處理。以下是簡要概述：

### 主要功能：

- 安裝必要套件：
  
	安裝 ultralytics 用於 YOLOv8：pip3 install ultralytics

	安裝 opencv-contrib-python 用於影像處理：pip install opencv-contrib-python

- 設定視窗及影像來源：

	cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL) 設定視窗名稱與顯示模式。

	影片來源可以是檔案、視訊鏡頭或網路影片。此範例使用 target = 'Video\Traffic_01.mp4' 來讀取本地影片檔。

- 載入 YOLO 模型：

	使用 YOLO('yolov8x.pt') 載入 YOLOv8 的權重檔案（可以選擇不同模型，這裡使用的是 yolov8x.pt，適用於多種物體識別任務）。

- 讀取影片及進行物體檢測：

	使用 OpenCV 讀取影片，每幀進行物體檢測：results = model(frame, verbose=False)。

	偵測到的物體會標註在影像上，並用 cv2.GaussianBlur 模糊處理物體區域。

- 顯示 FPS 並繪製結果：

	計算每秒幀數（FPS）並顯示在畫面上：cv2.putText(frame, 'FPS=' + str(FPS), (20, 35), ...)。

	使用 frame = results[0].plot() 來繪製檢測結果，並顯示在視窗中。

- 重播影片功能：

	如果影片播放完畢，會顯示提示訊息：cv2.putText(frame, '請按任意鍵, 將重新撥放', (20, 60), ...)。

	使用 cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 將影片回到第一幀，並繼續播放。

- 退出程式：

	按下 Esc 鍵 (key == 27) 可退出程式。

辨識結果，人、物品等
![image](https://github.com/user-attachments/assets/5c01a1d1-bde2-4e20-930c-6b148f603264)

辨識結果，車
![image](https://github.com/user-attachments/assets/13b05326-d6a2-4474-a7dd-5c6010fc4323)

## 3-2 01_ObjectDetectionToMarkBySelf.py

相較 01_ObjectDetection.py，顯示統計資訊，並對偵測到的物體進行邊框繪製框線、顯示屬性資訊、高斯模糊處理。

![image](https://github.com/user-attachments/assets/f455772b-ac09-44b9-adce-a65d514bea28)

------

# 4. Note

```diff
! 1.本程式所使用的 image 資料夾內女生模特兒皆為我使用SD繪製出來的虛擬人物。
! 2.本程式所使用的 Traffic_01.mp4 為路況即時影像取得。
```

------

## About Me
Thanks & Best Regards !

蔡承廷

​Senior Engineer of Semiconductor Product/Testing & ​Automation

Email: ​​kp924606@gmail.com

LinkedIn:https://www.linkedin.comin/tsai-cheng-ting/
