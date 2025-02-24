import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd

# ========================== CONFIGURATION ==========================
BASE_PATH = r'C:\Users\rusha\OneDrive\Desktop\TSP\ImagesAttendance'
ATTENDANCE_FILE = r'C:\Users\rusha\OneDrive\Desktop\TSP\Attendance.csv'
ATTENDANCE_GAP_SECONDS = 10

os.makedirs(BASE_PATH, exist_ok=True)

# ======================= FACE REGISTRATION ============================
def capture_face():
    name = input("Enter your name: ").strip()
    if not name:
        print("Invalid name. Try again.")
        return

    cap = cv2.VideoCapture(0)
    print("Capturing face... Press 'c' to capture, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error!")
            break

        cv2.imshow('Face Registration - Press "c" to capture', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            face_locations = face_recognition.face_locations(frame)
            if len(face_locations) == 0:
                print("No face detected. Try again.")
                continue

            file_path = os.path.join(BASE_PATH, f'{name}.jpg')
            cv2.imwrite(file_path, frame)
            print(f"Face captured and saved as {file_path}")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===================== ENCODINGS =============================
def load_and_encode_images():
    images = []
    class_names = []
    encodings = []

    for file_name in os.listdir(BASE_PATH):
        if file_name.endswith('.jpg'):
            img_path = os.path.join(BASE_PATH, file_name)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            encode = face_recognition.face_encodings(img_rgb)
            if encode:
                encodings.append(encode[0])
                class_names.append(os.path.splitext(file_name)[0])
            else:
                print(f"[WARNING] No face detected in {file_name}. Please re-register.")

    if not encodings:
        print("No valid images found. Please register at least one face.")

    return encodings, class_names

# ===================== ATTENDANCE MARKING =========================
def mark_attendance(name):
    date = datetime.now().strftime('%Y-%m-%d')
    time = datetime.now().strftime('%H:%M:%S')

    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Status'])
        df.to_csv(ATTENDANCE_FILE, index=False)

    df = pd.read_csv(ATTENDANCE_FILE)

    if not ((df['Name'] == name) & (df['Date'] == date)).any():
        new_entry = pd.DataFrame([[name, date, time, 'Present']], columns=['Name', 'Date', 'Time', 'Status'])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"{name} marked present at {time} on {date}")

# =================== SHOW ATTENDANCE =============================
def show_attendance():
    if not os.path.exists(ATTENDANCE_FILE):
        print("No attendance records found.")
        return

    df = pd.read_csv(ATTENDANCE_FILE)
    print("\nAttendance Records:")
    print(df.to_string(index=False))

# =================== FACE RECOGNITION =============================
def recognize_faces(encodings, class_names):
    cap = cv2.VideoCapture(0)
    last_seen = {}
    print("[INFO] Recognizing faces... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error!")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        current_time = datetime.now()

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(encodings, face_encoding)
            face_distances = face_recognition.face_distance(encodings, face_encoding)

            if any(matches):
                best_match_index = np.argmin(face_distances)
                name = class_names[best_match_index].upper()

                if name not in last_seen or (current_time - last_seen[name]).seconds > ATTENDANCE_GAP_SECONDS:
                    mark_attendance(name)
                    last_seen[name] = current_time

                y1, x2, y2, x1 = face_location
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Face Recognition - Attendance System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ========================== MAIN MENU ============================
while True:
    print("\nOptions: 1. Register New Face | 2. Start Attendance | 3. Show Attendance | 4. Exit")
    choice = input("Enter your choice (1/2/3/4): ").strip()

    if choice == '1':
        capture_face()
    elif choice == '2':
        encodings, class_names = load_and_encode_images()
        if encodings:
            recognize_faces(encodings, class_names)
        else:
            print("[ERROR] No face encodings available. Register faces first!")
    elif choice == '3':
        show_attendance()
    elif choice == '4':
        print("Exiting...")
        break
    else:
        print("Invalid choice. Please select a valid option.")
#==========================================END========================================================
