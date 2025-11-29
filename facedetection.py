import cv2 as cv
import winsound
import smtplib
from email.message import EmailMessage
from datetime import datetime
import os

# ---------- LOAD HAAR CASCADES ----------
face = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye  = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

if face.empty():
    print("Face cascade not loaded")
    exit()
else:
    print("Face model loaded")

# ---------- EMAIL FUNCTION ----------
def send_email(image_path):
    EMAIL = "arvindg12125@gmail.com"
    PASSWORD = "zrxcycicfsippzkk" # USE APP PASSWORD 
    TO = "aiuse12125@gmail.com"

    msg = EmailMessage()
    msg['Subject'] = " FACE DETECTED ALERT"
    msg['From'] = EMAIL
    msg['To'] = TO
    msg.set_content("Face detected. Screenshot attached.")

    with open(image_path, 'rb') as f:
        msg.add_attachment(f.read(), maintype='image', subtype='jpeg', filename=image_path)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL, PASSWORD)
        smtp.send_message(msg)

    print(" EMAIL SENT")

# ---------- MOBILE CAMERA URL ----------
cam = cv.VideoCapture(0)

if not cam.isOpened():
    print("Camera not accessible")
    exit()

sent = False
beeped = False

# ---------- LOOP ----------
while True:
    ret, frame = cam.read()
    if not ret:
        print("Cannot grab frame")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3,3), 2)

    faces = face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)
    real_faces = []

    # FILTER SMALL FALSE DETECTIONS
    for (x, y, w, h) in faces:
        if w > 100 and h > 100:
            real_faces.append((x,y,w,h))
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    # ---------- ALERT LOGIC ----------
    if len(real_faces) > 0:

        # BEEP ONCE
        if not beeped:
            winsound.Beep(2500, 300)
            beeped = True

        # EMAIL ONCE
        if not sent:
            os.makedirs("detected_faces", exist_ok=True)
            filename = f"detected_faces/detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv.imwrite(filename, frame)
            send_email(filename)
            sent = True
    else:
        beeped = False
        sent = False

    # DISPLAY WINDOW
    cv.imshow("Face Detection", frame)

    # ESC TO EXIT
    if cv.waitKey(1) == 27:
        break

# RELEASE
cam.release()
cv.destroyAllWindows()
