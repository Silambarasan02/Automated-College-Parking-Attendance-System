from ultralytics import YOLO
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

# Initialize PaddleOCR for number plate recognition
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Runs once to load OCR model

# Load the input image
img = cv2.imread("q1.jpg")  # Input image path

# Convert image from BGR to RGB for YOLO processing
new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perform object detection using the YOLO model
results = model(new_img, stream=True, device=device)

# Initialize dictionary and list for storing rider information
li = dict()
rider_box = []

# Open the file to write plate number and helmet status
with open("fi.txt", "w") as file:
    for r in results:
        boxes = r.boxes
        xy = boxes.xyxy  # Bounding box coordinates
        confidences = boxes.conf  # Confidence scores
        classes = boxes.cls  # Class indices

        # Combine bounding box coordinates, confidences, and class labels
        new_boxes = torch.cat((xy, confidences.unsqueeze(1), classes.unsqueeze(1)), 1)

        try:
            # Sort boxes by class labels for consistency
            new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]
            # Extract rows where class == 2 (rider)
            rider_indices = torch.where(new_boxes[:, -1] == 2)[0]
            rider_rows = new_boxes[rider_indices]

            # Store bounding boxes for riders
            for rider_box_data in rider_rows:
                x1, y1, x2, y2 = map(int, rider_box_data[:4])
                rider_box.append((x1, y1, x2, y2))

        except Exception as e:
            print(f"Error processing riders: {e}")

        for i, box in enumerate(new_boxes):
            x1, y1, x2, y2 = map(int, box[:4])  # Bounding box coordinates
            w, h = x2 - x1, y2 - y1  # Width and height of the bounding box
            conf = float(box[4].item())  # Confidence score
            cls = int(box[5].item())  # Class index

            # Filter detections based on class and confidence thresholds
            if (classNames[cls] == "without helmet" and conf >= 0.5) or \
                    (classNames[cls] == "rider" and conf >= 0.45) or \
                    (classNames[cls] == "number plate" and conf >= 0.5):

                if classNames[cls] == "rider":
                    rider_box.append((x1, y1, x2, y2))  # Append rider box if class is 'rider'

                if rider_box:
                    for j, rider in enumerate(rider_box):
                        # Check if the detected object is inside the rider's bounding box
                        if x1 + 10 >= rider[0] and y1 + 10 >= rider[1] and x2 <= rider[2] and y2 <= rider[3]:
                            # Draw bounding box and label for detected object
                            cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
                            cvzone.putTextRect(img, f"{classNames[cls].upper()}", (x1 + 10, y1 - 10), scale=1.5,
                                               offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))

                            # Keep track of objects detected for each rider
                            li.setdefault(f"rider{j}", []).append(classNames[cls])

                            # Process number plate if detected
                            if classNames[cls] == "number plate":
                                npx, npy, npw, nph = x1, y1, w, h
                                crop = img[npy:npy + nph, npx:npx + npw]

                        # If all required objects (rider, helmet, number plate) are detected
                        if len(set(li.get(f"rider{j}", []))) == 3:
                            try:
                                vehicle_number, confidence = predict_number_plate(crop, ocr)
                                if vehicle_number and confidence:
                                    # Display recognized vehicle number and confidence score
                                    cvzone.putTextRect(img, f"{vehicle_number} {round(confidence * 100, 2)}%",
                                                       (x1, y1 - 50), scale=1.5, offset=10,
                                                       thickness=2, colorT=(39, 40, 41), colorR=(105, 255, 255))

                                    # Get helmet status for the rider
                                    helmet_status = "with helmet" if "with helmet" in li[f"rider{j}"] else "without helmet"

                                    # Save plate number and helmet status to the file
                                    file.write(f"{vehicle_number} -> {helmet_status}\n")

                            except Exception as e:
                                print(f"OCR error: {e}")

# Save the output image with annotations
cv2.imwrite("output_image2.jpg", img)

# Display the output image
#cv2.imshow('Output Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
