import cv2
import imutils

# 1. Menginisialisasi detektor orang HOG
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 2. Membaca file video
cap = cv2.VideoCapture('r.mp4')

while cap.isOpened():
    # Membaca streaming video frame demi frame
    ret, image = cap.read()
    
    if ret:
        # Mengubah ukuran gambar/frame
        image = imutils.resize(image, width=min(400, image.shape[1]))
        
        # Mendeteksi pejalan kaki di dalam frame
        (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
        
        # Menggambar kotak merah di sekitar pejalan kaki
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
        # Menampilkan Gambar keluaran
        cv2.imshow("Image", image)
        
        # Tekan 'q' di keyboard untuk keluar dari video sebelum selesai
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Membersihkan proses setelah selesai
cap.release()
cv2.destroyAllWindows()
