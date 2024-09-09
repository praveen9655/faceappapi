from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
from PIL import Image
import io
import numpy as np
import pickle
import base64

app = Flask(__name__)
CORS(app)  # Allow CORS for all routes

# Load known face encodings and names
with open('trained_faces.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Simple welcome screen
@app.route('/')
def welcome():
    return "<h1>Welcome to the Face Recognition API</h1><p>Use the /api/identify endpoint to identify faces.</p>"

@app.route('/api/identify', methods=['POST'])
def identify_face():
    data = request.json
    image_data = data['image'].split(',')[1]  # Remove 'data:image/jpeg;base64,' prefix
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image_np = np.array(image)

    rgb_frame = image_np[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            return jsonify({'name': name})

    return jsonify({'name': 'Unknown'})

if __name__ == '__main__':
    app.run(debug=True)
