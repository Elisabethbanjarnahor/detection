import cv2
import imutils

# 1. Menginisialisasi detektor orang HOG
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 2. Membaca Gambar
image = cv2.imread('walk.jpg')

# 3. Mengubah ukuran gambar agar pemrosesan lebih ringan
image = imutils.resize(image, width=min(400, image.shape[1]))

# 4. Mendeteksi semua wilayah yang memiliki pejalan kaki di dalamnya
# Menggunakan parameter winStride, padding, dan scale dari PDF
(regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)

# 5. Menggambar kotak merah di setiap wilayah yang terdeteksi
for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 6. Menampilkan Gambar keluaran
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
