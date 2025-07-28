import cv2
import face_recognition

def compare_faces(img1, img2):
    try:
        rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        encodings1 = face_recognition.face_encodings(rgb_img1)
        encodings2 = face_recognition.face_encodings(rgb_img2)

        if not encodings1 or not encodings2:
            return False, 0.0

        face_distance = face_recognition.face_distance([encodings1[0]], encodings2[0])[0]
        similarity = 1 - face_distance
        return similarity > 0.5, similarity

    except Exception:
        return False, 0.0
