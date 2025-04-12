import cv2
import numpy as np
import os
from datetime import datetime

#  webcam
video_capture = cv2.VideoCapture(0)

# Create directories
if not os.path.exists('known_faces'):
    os.makedirs('known_faces')
if not os.path.exists('face_db'):
    os.makedirs('face_db')

# Load SFace model
model_path = "face_recognition_sface_2021dec.onnx"

config = ""
backend_id = cv2.dnn.DNN_BACKEND_OPENCV
target_id = cv2.dnn.DNN_TARGET_CPU

# face recognizer
recognizer = cv2.FaceRecognizerSF.create(model_path, config, backend_id, target_id)

# Thresholds
COSINE_THRESHOLD = 0.363
L2_THRESHOLD = 1.128


def load_known_faces():
    known_face_features = []
    known_face_names = []

    # Load each feature from the face_db directory
    for filename in os.listdir('face_db'):
        if filename.endswith('.npy'):
            feature = np.load(f'face_db/{filename}')
            known_face_features.append(feature)
            known_face_names.append(os.path.splitext(filename)[0])

    return known_face_features, known_face_names

def capture_new_face(name):
    print(f"Capturing new face for {name}...")
    ret, frame = video_capture.read()
    if ret:
        rgb_frame = frame[:, :, ::-1]
        # Detect
        face_detector = cv2.FaceDetectorYN.create(
            "face_detection_yunet_2023mar.onnx",
            "",
            (320, 320), 0.9, 0.3, 5000
        )
        #  size
        height, width, _ = frame.shape
        face_detector.setInputSize((width, height))
        # Detect faces
        _, faces = face_detector.detect(frame)
        if faces is not None and len(faces) == 1:
            aligned_face = recognizer.alignCrop(frame, faces[0])
            face_feature = recognizer.feature(aligned_face)
            # Save
            np.save(f'face_db/{name}.npy', face_feature)
            cv2.imwrite(f'known_faces/{name}.jpg', frame)
            print(f"Face saved as known_faces/{name}.jpg")
            return True
        else:
            print("Couldn't detect exactly one face in the frame.")
    return False


# Load known faces
known_face_features, known_face_names = load_known_faces()

# Initialize face detector
face_detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

print("Starting face recognition. Press 'q' to quit, 'n' to add new face.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    height, width, _ = frame.shape
    face_detector.setInputSize((width, height))

    # Detect
    _, faces = face_detector.detect(frame)

    if faces is not None:
        for face in faces:
            # Draw bounding box
            box = list(map(int, face[:4]))
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(frame, box, color, thickness, cv2.LINE_AA)

            landmarks = list(map(int, face[4:len(face) - 1]))
            landmarks = np.array_split(landmarks, len(landmarks) / 2)
            for landmark in landmarks:
                radius = 2
                thickness = -1
                cv2.circle(frame, landmark, radius, color, thickness, cv2.LINE_AA)

            aligned_face = recognizer.alignCrop(frame, face)

            face_feature = recognizer.feature(aligned_face)

            # Compare with known faces
            best_match_name = "Unknown"
            best_match_score = 0

            for known_feature, known_name in zip(known_face_features, known_face_names):
                score = recognizer.match(face_feature, known_feature, cv2.FaceRecognizerSF_FR_COSINE)

                if score > best_match_score and score >= COSINE_THRESHOLD:
                    best_match_score = score
                    best_match_name = known_name

            # Display name
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.8
            thickness = 2
            cv2.putText(frame, f"{best_match_name} ({best_match_score:.2f})", position, font, scale, color, thickness,
                        cv2.LINE_AA)
    # Display the resulting image
    cv2.imshow('Face Recognition', frame)
    # Check for key presses
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('n'):
        name = input("Enter name for new face: ")
        if capture_new_face(name):
            known_face_features, known_face_names = load_known_faces()
# Release resources
video_capture.release()
cv2.destroyAllWindows()