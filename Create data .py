import cv2
import os
import numpy as np
detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
id=1
if id == 1:
    print(0)
    out_dir = 'd:/TriTueNhanTao/NhanDienKhuonMat/datasetIM'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    num_people = 2  # Số người
    num_images = 20 # Số ảnh mỗi người
    for i in range(1, num_people+1):
        for j in range(1, num_images+1):
            filename = f'd:/TriTueNhanTao/NhanDienKhuonMat/data/anh.{i}.{j}.jpg'
            frame = cv2.imread(filename)
            if frame is None:
                print(f"Không đọc được {filename}")
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fa = detector.detectMultiScale(gray, 1.1, 5)
            if len(fa) == 0:
                print(f"Không phát hiện khuôn mặt trong {filename}")
                continue
            for(x,y,w,h) in fa:
                cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)
                cv2.imwrite(os.path.join(out_dir, f'anh{i}.{j}.jpg'), gray[y:y+h,x:x+w])
                
if id == 2:
    out_dir = 'd:/TriTueNhanTao/NhanDienKhuonMat/datasetWB'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    num_people = 1  # Số người
    num_images = 20 # Số ảnh mỗi người
    cap = cv2.VideoCapture(0)
    detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    for i in range(1, num_people+1):
        sampleNum = 0
        print(f"Bắt đầu chụp cho người thứ {i}")
        while sampleNum < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Không lấy được khung hình từ webcam")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fa = detector.detectMultiScale(gray, 1.1, 5)
            for(x,y,w,h) in fa:
                cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)
                sampleNum+=1
                cv2.imwrite(os.path.join(out_dir, f'anh{i}.{sampleNum}.jpg'), gray[y:y+h,x:x+w])
                print(f"Đã lưu ảnh {sampleNum} cho người {i}")
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()