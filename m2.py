from ultralytics import YOLO
import math
import cv2
import cvzone
import torch
from image_to_text import predict_number_plate
from paddleocr import PaddleOCR

# Load the YOLO model with the trained weights
model = YOLO(r"E:/django-student-attendance-system-master/runs/detect/train/weights/best.pt")

# Set the device to CPU (or 'cuda' for GPU if available)
device = torch.device("cpu")  # Use 'cuda' for GPU support

# Define class names corresponding to model predictions
classNames = ["with helmet", "without helmet", "rider", "number plate"]
num = 0
old_npconf = 0

# Initialize PaddleOCR for number plate recognition
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Runs once to load OCR model

# Open the video file
cap = cv2.VideoCapture("videos/22.mp4")

# Get frame dimensions and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize the video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Open the file for writing plate number and helmet status
with open("files.txt", "w") as f:
    
    while True:
        success, img = cap.read()
        if not success:
            break
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(new_img, stream=True, device="mps")
        
        for r in results:
            boxes = r.boxes
            li = dict()
            rider_box = list()
            xy = boxes.xyxy
            confidences = boxes.conf
            classes = boxes.cls
            new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)
            try:
                new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]
                indices = torch.where(new_boxes[:, -1] == 2)
                rows = new_boxes[indices]
                for box in rows:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    rider_box.append((x1, y1, x2, y2))
            except:
                pass

            for i, box in enumerate(new_boxes):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box[4] * 100)) / 100
                cls = int(box[5])

                if classNames[cls] in ["without helmet", "rider", "number plate"] and conf >= 0.5:
                    if classNames[cls] == "rider":
                        rider_box.append((x1, y1, x2, y2))

                    if rider_box:
                        for j, rider in enumerate(rider_box):
                            if x1 + 10 >= rider_box[j][0] and y1 + 10 >= rider_box[j][1] and x2 <= rider_box[j][2] and y2 <= rider_box[j][3]:
                                cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
                                cvzone.putTextRect(img, f"{classNames[cls].upper()}", (x1 + 10, y1 - 10), scale=1.5,
                                                   offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))

                                li.setdefault(f"rider{j}", []).append(classNames[cls])
                                if classNames[cls] == "number plate":
                                    npx, npy, npw, nph = x1, y1, w, h
                                    crop = img[npy:npy + h, npx:npx + w]

                            if len(set(li.get(f"rider{j}", []))) == 3:  # Rider, helmet, and number plate detected
                                try:
                                    vehicle_number, confidence = predict_number_plate(crop, ocr)
                                    if vehicle_number and confidence:
                                        helmet_status = "with helmet" if "with helmet" in li[f"rider{j}"] else "without helmet"
                                        
                                        # Write plate number and helmet status to the file
                                        f.write(f"{vehicle_number} -> {helmet_status}\n")

                                        cvzone.putTextRect(img, f"{vehicle_number} {round(confidence * 100, 2)}%",
                                                           (x1, y1 - 50), scale=1.5, offset=10,
                                                           thickness=2, colorT=(39, 40, 41),
                                                           colorR=(105, 255, 255))
                                except Exception as e:
                                    print(f"OCR error: {e}")
        
        # Write the processed frame to the output video
        output.write(img)
        
        # Display the frame
        cv2.imshow('Video', img)

        # Exit the program if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
output.release()
cv2.destroyAllWindows()
