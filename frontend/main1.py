
import os
import cv2

from flask import Flask, render_template, Response

app = Flask(__name__, instance_relative_config=True)



@app.route('/hello')
def test(test_config=None):
 
    return 'Hello, World!'
        
        
        
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Return the frame as byte data in a response with proper headers for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    
                    
@app.route('/index')
def login():
    return render_template("index.html")



if __name__ == '__main__':
   app.run(debug=True)

