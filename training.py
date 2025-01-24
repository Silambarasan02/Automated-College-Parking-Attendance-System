from ultralytics import YOLO

# yolo model creation
#model = YOLO("E:/Allproject/Real-Time-Detection-of-Helmet-Violations-and-Capturing-Bike-Numbers-from-Number-Plates-main/yolo-weights/yolov8n.pt")

model = YOLO(r"E:\Allproject\Real-Time-Detection-of-Helmet-Violations-and-Capturing-Bike-Numbers-from-Number-Plates-main\sample\yolov8n.pt")
print("Model is load")
model.train(data="coco128.yaml", imgsz=320, batch=4, epochs=20, workers=0)
