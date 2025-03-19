#The following code is based on the "AI Application Development Course" and has been partially modified after further personal research.
#Version=3.11.11

#1.安裝 YOLOv8
#pip3 install ultralytics

#2.安裝 opencv
#pip install opencv-contrib-python

from ultralytics import YOLO
import cv2,time

#設定視窗名稱及型態
cv2.namedWindow('YOLOv8', cv2.WINDOW_NORMAL)

#影像來源變數
#0 視訊鏡頭
#檔名.mp4 取檔案, 您事先錄製好的影片
#網址 取影像,可搭配使用臺灣即時影像:tw.live
#圖片, 請用jpg避免格式不同圖取錯誤
#target=0
#target=fr"01_image\02.jpg"
target='Video\Traffic_01.mp4'
#target='https://YourVideo'

# n,s,m,l,x 五種大小
#yolov8n-seg.pt   物件分割 (Segmentation)     汽車、人、物品輪廓分割
#yolov8n-pose.pt  姿勢估計 (Pose Estimation)  人體關鍵點偵測 (如手腳位置)
#yolov8n-cls.pt   影像分類 (Classification)   辨識貓、狗、車等類別
model = YOLO('yolov8x.pt')
#model = YOLO('yolov8l-pose.pt')  # n,s,m,l,x 五種大小, pose 身體模型
#model = YOLO('yolov8l-seg.pt')  # n,s,m,l,x 五種大小, pose 身體模型
names=model.names
print('模型名稱:', names)
#{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

cap=cv2.VideoCapture(target)

while 1:
    st=time.time()  
    r,frame = cap.read()
    if r==False:
        cv2.putText(frame, '請按任意鍵, 將重新撥放', (20, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)  
        print("請按任意鍵, 將重新撥放")
        cv2.waitKey(0)  # 等待使用者按下任意鍵
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重新設置回第一幀
        continue  # 繼續播放
        #break

    #執行模型辨識
    results = model(frame,verbose = False)

    #一個一個解析辨識結果
    carCount = 0
    for box in results[0].boxes.data:
        #tensor([622.4321, 203.1665, 701.6540, 243.0457,   0.8392,   2.0000], device='cuda:0')
        x1 = int(box[0]) #X1 左
        y1 = int(box[1]) #Y1 上
        x2 = int(box[2]) #X2 右
        y2 = int(box[3]) #Y2 下
        r = int(box[4] * 100) #信任度
        n = model.names[int(box[5])] #物件名稱

        if n in ['person', 'bicycle', 'car', 'bus', 'truck', 'motorcycle', 'traffic light']:
            carCount = carCount + 1
            # 畫外框       照片     左上     右下    BGR      外框
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
    
            #一律都進行模糊處理
            roi = frame[y1:y2, x1:x2]  # 擷取區域
            blurred = cv2.GaussianBlur(roi, (51, 51), 30)  # 高斯模糊
            frame[y1:y2, x1:x2] = blurred  # 取代原始區域

            #寫名字
            cv2.putText(frame, n, (x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),1)    
   
    cv2.rectangle(frame, (20,20), (150,100), (255,255,255),-1)
    cv2.putText(frame, 'Total:' + str(carCount), (20,80),cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0),2) 

    #繪製偵測結果
    #frame= results[0].plot()
    et=time.time()
   
    FPS=int(1/(et-st)) #評估時間
    cv2.putText(frame, 'FPS=' + str(FPS), (20, 35), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)    
    cv2.imshow('YOLOv8', frame)
    
    key=cv2.waitKey(1)
    #退出程式
    if key==27:
        break
