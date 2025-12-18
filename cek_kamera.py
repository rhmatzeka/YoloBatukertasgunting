import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO(r'C:\runs\classify\rps_final_success\weights\best.pt')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    # Konversi ke ruang warna HSV (lebih baik untuk deteksi warna kulit)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Range warna kulit manusia (umum)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Membuat mask/topeng warna kulit
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # Membersihkan bintik-bintik kecil (noise)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Cari objek warna kulit terbesar
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 3000:
            x, y, w_b, h_b = cv2.boundingRect(c)
            
            # Buat kotak persegi (Square ROI) agar sesuai input YOLO
            side = max(w_b, h_b) + 40
            cx, cy = x + w_b//2, y + h_b//2
            x1, y1 = max(0, cx - side//2), max(0, cy - side//2)
            x2, y2 = min(frame.shape[1], x1 + side), min(frame.shape[0], y1 + side)

            roi = frame[y1:y2, x1:x2]
            
            if roi.size != 0:
                results = model.predict(source=roi, verbose=False)
                prob = results[0].probs
                label = results[0].names[prob.top1]
                conf = prob.top1conf.item()

                # Visualisasi
                color = (0, 255, 0) if conf > 0.7 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2%}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Deteksi Kulit (Lebih Stabil)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()