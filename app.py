from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
from ultralytics import YOLO
import pyttsx3

app = Flask(__name__)

# Load your YOLO model
model = YOLO("best.pt")  # Make sure this file exists in your folder

# Text-to-speech
engine = pyttsx3.init()

# English to Arabic mapping (customize as needed)
arabic_map = {
    'A': 'ا', 'B': 'ب', 'C': 'ج', 'D': 'د', 'E': 'ه',
    'F': 'ف', 'G': 'ج', 'H': 'ح', 'I': 'ي', 'J': 'ج',
    'K': 'ك', 'L': 'ل', 'M': 'م', 'N': 'ن', 'O': 'و',
    'P': 'ب', 'Q': 'ق', 'R': 'ر', 'S': 'س', 'T': 'ت',
    'U': 'ع', 'V': 'ف', 'W': 'و', 'X': 'كس', 'Y': 'ي', 'Z': 'ز'
}

detected_letters = []

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result
            label_en = results.names[int(cls)].upper()
            label_ar = arabic_map.get(label_en, label_en)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label_ar, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if not detected_letters or detected_letters[-1] != label_ar:
                detected_letters.append(label_ar)
                engine.say(label_ar)
                engine.runAndWait()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', letters=''.join(detected_letters))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear', methods=['POST'])
def clear_letters():
    detected_letters.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
