from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import string
import time

# Load the trained model
model = load_model('model99.h5')

# Define class labels (A-Z and 1-9)
class_labels = list(string.ascii_uppercase) + [str(i) for i in range(1, 10)]

# Initialize Flask application
app = Flask(__name__)

# Global variable to hold the captured sequence of letters
captured_sequence = ""
last_predicted_letter = ""
prediction_confidence_threshold = 0.5  # Set a confidence threshold

def preprocess_image(img, target_size=(128, 128)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img_array = img / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_letter(frame):
    global last_predicted_letter
    img_array = preprocess_image(frame)
    predictions = model.predict(img_array)
    
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)  # Get the confidence of the prediction
    
    if confidence > prediction_confidence_threshold:
        predicted_label = class_labels[predicted_class[0]]
        
        # Only add to sequence if it's different from the last prediction
        if predicted_label != last_predicted_letter:
            last_predicted_letter = predicted_label
            return predicted_label
    return None

def gen_frames():
    global captured_sequence
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            predicted_letter = predict_letter(frame)
            if predicted_letter:
                captured_sequence += predicted_letter  # Append to the sequence

            # Display the prediction on the frame
            cv2.putText(frame, f"Predicted: {predicted_letter or last_predicted_letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Sequence: {captured_sequence}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Encode the frame for display
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset_sequence', methods=['POST'])
def reset_sequence():
    global captured_sequence
    captured_sequence = ""  # Reset the sequence
    return jsonify(success=True)

@app.route('/get_sequence')
def get_sequence():
    return jsonify(sequence=captured_sequence)

if __name__ == "__main__":
    app.run(debug=True)
