import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x.pt")

cap = cv2.VideoCapture(0)
# Perform object detection on an image

if not  cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret,frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#results = model("https://ultralytics.com/images/bus.jpg")
#results[0].show()