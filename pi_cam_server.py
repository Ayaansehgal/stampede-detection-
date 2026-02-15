import cv2
import subprocess
import time
from flask import Flask, Response

app = Flask(__name__)

def generate_frames():

    try:
        cmd = [
            "rpicam-vid", "-t", "0", "--inline", "--width", "640", "--height", "480",
            "--framerate", "15", "--codec", "mjpeg", "-n", "-o", "-"
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

        buffer = b''

        start_time = time.time()
        while time.time() - start_time < 2:
            chunk = process.stdout.read(4096)
            if not chunk: break
            buffer += chunk
            a = buffer.find(b'\xff\xd8')
            b = buffer.find(b'\xff\xd9')
            if a != -1 and b != -1:
                print("Streaming via rpicam-vid!")

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer[a:b+2] + b'\r\n')
                buffer = buffer[b+2:]
                while True:
                    chunk = process.stdout.read(4096)
                    if not chunk: break
                    buffer += chunk
                    a = buffer.find(b'\xff\xd8')
                    b = buffer.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = buffer[a:b+2]
                        buffer = buffer[b+2:]
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        process.terminate()
    except Exception as e:
        print(f"rpicam-vid failed: {e}")

    print("Falling back to OpenCV...")
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def stream():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting Bulletproof Pi Streamer...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
