import os
import cv2

# Mengurangi log dari TensorFlow (opsional jika TensorFlow tidak digunakan)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Membuka kamera (0 untuk kamera utama)
cap = cv2.VideoCapture(0)

# Cek apakah kamera berhasil dibuka
if not cap.isOpened():
    print("Tidak dapat membuka kamera.")
    exit()

# Memuat model deteksi wajah dari OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera.")
        break

    # Konversi frame ke grayscale untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Menggambar frame di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Menampilkan frame dengan kotak di sekitar wajah
    cv2.imshow('Face Detector', frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepas resource kamera
cap.release()
cv2.destroyAllWindows()
