import cv2
import mediapipe as mp

# Inisialisasi MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2)

# Fungsi untuk menghitung jumlah jari yang terangkat
def count_fingers(hand_landmarks, hand_label):
    fingers = []
    
    # Thumb (Ibu Jari)
    if hand_label == "Right":
        # Jika tangan kanan, ibu jari dianggap terangkat jika landmark THUMB_TIP lebih kiri dari THUMB_IP
        if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
            fingers.append(1)  # Thumb is up
        else:
            fingers.append(0)
    else:
        # Jika tangan kiri, ibu jari dianggap terangkat jika landmark THUMB_TIP lebih kanan dari THUMB_IP
        if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
            fingers.append(1)  # Thumb is up
        else:
            fingers.append(0)

    # Index, Middle, Ring, Pinky Fingers
    for finger_tip_id, finger_dip_id in zip(
        [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
         mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP],
        [mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
         mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.PINKY_DIP]):
        
        if hand_landmarks.landmark[finger_tip_id].y < hand_landmarks.landmark[finger_dip_id].y:
            fingers.append(1)  # Finger is up
        else:
            fingers.append(0)

    return fingers.count(1)  # Mengembalikan jumlah jari yang terangkat


# Buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membuka kamera.")
        break

    # Konversi gambar ke RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Proses deteksi tangan
    results = hands.process(image)

    # Gambar tangan pada frame
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Gambar koneksi landmarks pada tangan
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Mendapatkan informasi label tangan (kanan/kiri)
            hand_label = results.multi_handedness[idx].classification[0].label

            # Menghitung jumlah jari yang terangkat untuk masing-masing tangan
            fingers_up = count_fingers(hand_landmarks, hand_label)
            cv2.putText(image, f'{hand_label} hand: {fingers_up} fingers', (10, 50 + idx * 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow('Hand Tracking', image)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Lepaskan resources
cap.release()
cv2.destroyAllWindows()
