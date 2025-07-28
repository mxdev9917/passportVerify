from ultralytics import YOLO

model = YOLO("best.pt")
REQUIRED_CLASSES = ["Passport", "Photo", "MRZ"]
