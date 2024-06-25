from flask import Flask, render_template, Response, request
import cv2
import math
from ultralytics import YOLO
import numpy as np
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import threading

app = Flask(__name__)
# Load YOLO model
model = YOLO("C:/Kaavya/New/best.pt")
classNames = ['drone']
output_video_path = [None]

def send_email_with_attachment(video_path, to_email):
    fromadd = "k6785291@gmail.com"  # Update with your email address
    password = "ryro ltsn pidm mqlu"  # Update with your email password
    
    toadd = to_email  # Email address obtained from HTML form

    msg = MIMEMultipart()
    msg['From'] = fromadd
    msg['To'] = toadd
    msg['Subject'] = "Drone Detected"

    filename = "drone_video.mp4"
    attachment = open(video_path, "rb")

    p = MIMEBase('application', 'octet-stream')
    p.set_payload(attachment.read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', f"attachment; filename= {filename}")
    msg.attach(p)

    # SMTP Configuration
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(fromadd, password)
        text = msg.as_string()
        server.sendmail(fromadd, toadd, text)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Error: {str(e)}")

def process_video(video_path, to_email=None):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if video_path == 0:
        fps = 5
    
    def gen_frames():
        if output_video_path[0] is None:
            output_video_path[0] = 'drone.mp4'

        video_limit = 10 * 1024 * 1024
        current_file_size = 0
        file_counter = 1
        out = cv2.VideoWriter(output_video_path[0], cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        try:
            while True:
                success, img = cap.read()
                if not success:
                    break
                
                # Detect objects using YOLO
                results = model(img, stream=True)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        class_name = classNames[cls]
                        label = f' {class_name}{conf} '
                        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                        c2 = x1 + t_size[0], y1 - t_size[1] - 3
                        if class_name == 'drone':
                            color = (0, 204, 255)
                        else:
                            color = (85, 45, 255)
                        if conf > 0.5:
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                            out.write(img)
                            current_file_size = out.get(cv2.CAP_PROP_POS_MSEC)
                            if current_file_size > video_limit:
                                out.release()
                                if to_email:
                                    threading.Thread(target=send_email_with_attachment, args=(output_video_path[0], to_email)).start()
                                output_video_path[0] = f"{output_video_path}_{file_counter}.mp4"
                                current_file_size = 0
                                file_counter += 1
                                out = cv2.VideoWriter(output_video_path[0], cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        finally:
            out.release()
            if to_email:
                threading.Thread(target=send_email_with_attachment, args=(output_video_path[0], to_email)).start()
            output_video_path[0] = f"{output_video_path}_{file_counter}.mp4"

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('awesome.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    to_email = request.form.get('email')
    # Get the uploaded video file
    video_file = request.files['video']
    # Save the uploaded video to a temporary location
    video_path = "templates/video.mp4"
    video_file.save(video_path)
    if to_email:
        return process_video(video_path, to_email)
    else:
        return process_video(video_path)

@app.route('/video_feed_webcam', methods=['POST'])
def video_feed_webcam():
    to_email = request.form.get('email')
    if to_email:
        return process_video(0, to_email)
    else:
        return process_video(0)

if __name__ == "__main__":
    app.run(debug=True)
