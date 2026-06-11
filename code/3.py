import cv2
import pytesseract
import matplotlib.pyplot as plt

# Jalur ke tesseract yang dapat dieksekusi
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_plate_number(image_path):
    # Muat gambar
    image = cv2.imread(image_path)
    
    # Konversi ke skala abu-abu (Grayscale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Terapkan Gaussian Blur untuk menghilangkan noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Deteksi tepi (Canny) untuk menyorot kontur pelat
    edges = cv2.Canny(blurred, 100, 200)
    
    # Temukan kontur untuk menemukan plat nomor
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Urutkan kontur berdasarkan area (urutan menurun dari yang terbesar)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    plate_contour = None
    for contour in contours:
        # Perkiraan kontur ke poligon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Periksa apakah kontur memiliki 4 simpul (persegi panjang, khas untuk pelat)
        if len(approx) == 4:
            plate_contour = approx
            break
            
    if plate_contour is not None:
        # Dapatkan koordinat kotak pembatas pelat nomor
        x, y, w, h = cv2.boundingRect(plate_contour)
        
        # Potong area pelat nomor (Grayscale untuk OCR, Berwarna untuk Zoom)
        plate_image = gray[y:y + h, x:x + w]
        color_plate = image[y:y + h, x:x + w] # <-- Mengambil potongan berwarna untuk zoom
        
        # Gambar kotak pembatas hijau di gambar asli
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Terapkan ambang batas (Thresholding) untuk membinarisasi area pelat
        _, thresh = cv2.threshold(plate_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # ==========================================================
        # BAGIAN ZOOM: Menampilkan Gambar Utama dan Zoom Berdampingan
        # ==========================================================
        plt.figure(figsize=(12, 6)) # Mengatur ukuran jendela plot
        
        # Posisi 1 (Kiri): Gambar Mobil Utuh + Kotak Hijau
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("1. Posisi Pelat Nomor")
        plt.axis('off')
        
        # Posisi 2 (Kanan): Hasil Crop / Zoom Pelat Nomor
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(color_plate, cv2.COLOR_BGR2RGB))
        plt.title("2. Hasil Zoom Pelat Nomor")
        plt.axis('off')
        
        plt.show()
        # ==========================================================
        
        # Lakukan OCR pada area pelat yang terdeteksi
        plate_number = pytesseract.image_to_string(thresh, config='--psm 8')
        return plate_number.strip()
    else:
        return "License plate not detected"

# Berikan jalur gambar
image_path = "car.jpg" # Ganti dengan jalur gambar Anda

# Deteksi dan cetak nomor plat
plate_number = detect_plate_number(image_path)
print("Detected Plate Number:", plate_number)
