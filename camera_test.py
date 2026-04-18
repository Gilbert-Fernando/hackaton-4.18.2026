#cd main.py
#python -m venv venv
#venv\Scripts\activate
#pip install opencv-python
#pip freeze > requirements.txt
#deactivate




import cv2

# Load the face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open the webcam
camera = cv2.VideoCapture(0)

while True:
    # Capture frame
    ret, frame = camera.read()

    if not ret:
        break

    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        print("Face detected!")

    # Show camera feed
    cv2.imshow("Camera", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
