import cv2
import mediapipe as mp
import time
import math

# --- Настройка камер и распознавания ---
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("Face.xml")

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

# индексы точек глаз (mediapipe face mesh)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [263, 387, 385, 362, 380, 373]

# функция вычисления EAR
def EAR(eye_points, landmarks):
    # eye_points - индексы точек
    # landmarks - точная карта лица
    p1 = landmarks[eye_points[0]]
    p2 = landmarks[eye_points[1]]
    p3 = landmarks[eye_points[2]]
    p4 = landmarks[eye_points[3]]
    p5 = landmarks[eye_points[4]]
    p6 = landmarks[eye_points[5]]

    # расстояния между точками
    A = math.dist(p2, p6)
    B = math.dist(p3, p5)
    C = math.dist(p1, p4)

    ear = (A + B) / (2.0 * C)
    return ear

blink_counter = 0
last_blink_time = time.time()

while True:
    success, img = video.read()
    if not success:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- детекция лиц твоим каскадом ---
    faces = face_cascade.detectMultiScale(gray, 1.55, 4)

    # отображаем прямоугольники
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # --- Обработка медиапайпом ---
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        # получаем 468 точек -> список (x,y)
        h, w, _ = img.shape
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append((lm.x * w, lm.y * h))

        # считаем EAR для обоих глаз
        left_ear = EAR(LEFT_EYE, landmarks)
        right_ear = EAR(RIGHT_EYE, landmarks)
        ear = (left_ear + right_ear) / 2

        # порог закрытого глаза
        if ear < 0.23:  
            blink_counter += 1
            last_blink_time = time.time()

        # если не моргал более 8 секунд → подозрительно
        if time.time() - last_blink_time > 8:
            cv2.putText(img, "FAKE FACE", (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)
        else:
            cv2.putText(img, "REAL FACE", (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break