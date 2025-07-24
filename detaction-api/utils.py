import cv2
import face_recognition
import os
import re

def allowed_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))

def save_image(image, folder, filename):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    cv2.imwrite(path, image)

def match_faces(face1_img, face2_img):
    try:
        face1_enc = face_recognition.face_encodings(face1_img)[0]
        face2_enc = face_recognition.face_encodings(face2_img)[0]
        return face_recognition.compare_faces([face1_enc], face2_enc)[0]
    except IndexError:
        return False

def extract_mrz_data(text):
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 0]
    if len(lines) < 2:
        return None

    first, second = lines[-2], lines[-1]
    if len(first) != len(second):
        return None

    data = {}
    try:
        data["document_type"] = first[0]
        data["issuing_country"] = first[2:5]
        data["last_name"], data["first_name"] = parse_names(first[5:])
        data["passport_number"] = second[0:9].replace("<", "")
        data["nationality"] = second[10:13]
        data["birth_date"] = second[13:19]
        data["sex"] = second[20]
        data["expiry_date"] = second[21:27]
    except Exception:
        return None

    return data

def parse_names(name_str):
    parts = name_str.split("<<")
    last = parts[0].replace("<", " ")
    first = parts[1].replace("<", " ") if len(parts) > 1 else ""
    return last.strip(), first.strip()
