import cv2 as cv
import winsound
import smtplib
from email.message import EmailMessage
from datetime import datetime
import os
import time

# Load face detector
face = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Email details
EMAIL = "here enter email from where you have to send"
PASSWORD = "APP Password enter here"   
TO = "here enter to whom you want to send this email"

def send_email(img):
    msg = EmailMessage()
    msg['Subject'] = "PERSON DETECTED"
    msg['From'] = EMAIL
    msg['To'] = TO
    msg.set_content("New person detected by camera.")

    with open(img, 'rb') as f:
        msg.add_attachment(f.read(), maintype='image', subtype='jpeg', filename=img)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(EMAIL, PASSWORD)
        s.send_message(msg)

# Camera
cam = cv.VideoCapture(0)
os.makedirs("persons", exist_ok=True)

sent_faces = {}   # stores/remember who already got emailed
WAIT = 10         # 10 sec takes before resending same person

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.2, 5)
    now = time.time()

    for (x,y,w,h) in faces:

        if w < 80 or h < 80:
            continue

        # Unique ID from face location
        key = f"{x}_{y}_{w}_{h}"

        # skip if already emailed recently
        if key in sent_faces and now - sent_faces[key] < WAIT:
            continue

        sent_faces[key] = now

        # Approximate body
        body = frame[y:y+6*h, max(0,x-w//2):x+w+w//2]

        if body.size == 0:
            continue

        filename = f"persons/person_{datetime.now().strftime('%H%M%S_%f')}.jpg"
        cv.imwrite(filename, body)
        

        winsound.Beep(2200, 200)
        send_email(filename)

        # Draw boxes
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv.rectangle(frame,(max(0,x-w//2),y),(x+w+w//2,y+6*h),(255,0,0),2)
        cv.putText(frame, "EMAIL SENT", (x,y-10),cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv.imshow("Multi Person Email System", frame)

    if cv.waitKey(1) == 27:
        break

cam.release()
cv.destroyAllWindows()
