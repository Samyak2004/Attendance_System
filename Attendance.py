import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Set your phone's camera IP and port
PHONE_IP = '1'  # Change this to your phone's IP
PORT = '4747'  # Change this to your app's port

# Construct video stream URL
PHONE_CAMERA_URL = f'http://{PHONE_IP}:{PORT}/video'

# Paths
path = r'C:\Users\samya\Documents\Tech_Saksham\Attendance_System\attendance_images'
attendance_file = r'C:\Users\samya\Documents\Tech_Saksham\Attendance_System\attendance.csv'

if not os.path.exists(path):
    os.makedirs(path)

if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write("Name,Date,Time\n")

# ======================= FACE REGISTRATION ============================
def capture_face():
    name = input("Enter your name: ").strip()
    cap = cv2.VideoCapture(PHONE_CAMERA_URL)  # Using phone camera
    print("Capturing face... Press 'c' to capture.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to fetch frame. Check camera connection!")
            break

        cv2.imshow('Face Registration - Press "c" to capture', frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            file_path = os.path.join(path, f'{name}.jpg')
            cv2.imwrite(file_path, frame)
            print(f"Image saved as {file_path}")
            break

    cap.release()
    cv2.destroyAllWindows()

# ===================== ENCODINGS =============================
def load_and_encode_images():
    images = []
    classNames = []
    for cl in os.listdir(path):
        curImg = cv2.imread(os.path.join(path, cl))
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])

    encodeList = []
    for img, name in zip(images, classNames):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
        else:
            print(f"Face not detected in {name}, please check the image.")
            classNames.remove(name)
    return encodeList, classNames

# ===================== ATTENDANCE MARKING =========================
def mark_attendance(name):
    date = datetime.now().strftime('%Y-%m-%d')

    try:
        with open(attendance_file, 'r+') as f:
            dataList = f.readlines()
            nameList = [entry.split(',')[0] for entry in dataList]

            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.write(f'{name},{date},{dtString}\n')
                print(f"{name} marked present at {dtString} on {date}")
    except PermissionError:
        print(f"Error: Permission denied for file {attendance_file}. Try running the script as administrator.")

# =================== FACE RECOGNITION =============================
def recognize_faces(encodeListKnown, classNames):
    cap = cv2.VideoCapture(PHONE_CAMERA_URL)  # Use IP-based camera

    print("Press 'q' to quit attendance system.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to fetch frame. Check camera connection!")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            if len(faceDis) > 0:
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()

                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                    mark_attendance(name)

        cv2.imshow('Attendance System', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ========================== MAIN MENU ============================
while True:
    print("\n1. Register New Face")
    print("2. Start Attendance System")
    print("3. Exit")

    choice = input("Enter your choice: ")

    if choice == '1':
        capture_face()

    elif choice == '2':
        encodeListKnown, classNames = load_and_encode_images()
        print(f"Encoding Complete for {len(classNames)} faces.")
        recognize_faces(encodeListKnown, classNames)

    elif choice == '3':
        print("Exiting...")
        break

    else:
        print("Invalid choice. Try again.")
