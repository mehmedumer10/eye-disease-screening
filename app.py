from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import requests
from io import BytesIO
import os

app = Flask(__name__)

# Load the trained model
# Path to the trained Keras model. Train it with train_cnn.ipynb, or set EYE_MODEL_PATH.
MODEL_PATH = os.environ.get('EYE_MODEL_PATH', os.path.join('models', 'Eyesprediction.h5'))
model = load_model(MODEL_PATH, compile=False)

# Class labels
CLASS_NAMES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# HTML Template with Local Background Video
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Eye Disease Detection</title>
    <style>
        body { 
            font-family: Arial; 
            margin: 0; 
            padding: 0; 
            overflow: hidden; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            height: 100vh; 
            color: #fff; 
        }
        #video-background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            z-index: -1;
        }
        .content {
            background: rgba(0, 0, 0, 0.7);
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
            text-align: center;
            max-width: 600px;
            width: 100%;
        }
        h2, h3 { 
            color: #fff; 
            margin-bottom: 20px; 
        }
        form { 
            margin-bottom: 20px; 
        }
        input[type=file], input[type=text], input[type=submit] {
            width: 100%; 
            padding: 10px; 
            margin-bottom: 15px; 
            border: none; 
            border-radius: 5px; 
        }
        input[type=submit] {
            background: #007bff;
            color: #fff;
            cursor: pointer;
        }
        input[type=submit]:hover {
            background: #0056b3;
        }
        ul { 
            list-style: none; 
            padding: 0; 
        }
        li { 
            background: rgba(255, 255, 255, 0.1); 
            padding: 10px; 
            margin-bottom: 10px; 
            border-radius: 5px; 
        }
        .error {
            color: #ff4444;
            font-weight: bold;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <!-- Background Video -->
    <video id="video-background" autoplay muted loop>
        <source src="{{ url_for('static', filename='bg.mp4') }}" type="video/mp4">
        Your browser does not support the video tag.
    </video>

    <!-- Content -->
    <div class="content">
        <h2>Eye Disease Detection</h2>
        <form method="POST" enctype="multipart/form-data">
            <label>Upload Eye Image:</label>
            <input type="file" name="image"><br>
            <label>Or enter image URL:</label>
            <input type="text" name="image_url" placeholder="https://example.com/image.jpg"><br>
            <input type="submit" value="Predict">
        </form>

        {% if error_message %}
            <div class="error">{{ error_message }}</div>
        {% endif %}

        {% if predictions %}
            <h3>Prediction Results:</h3>
            <ul>
            {% for row in predictions %}
                <li><strong>{{ row.class_name }}</strong>: {{ '{:.2f}%'.format(row.probability * 100) }}</li>
            {% endfor %}
            </ul>
        {% endif %}
    </div>
</body>
</html>
'''

# Image Preprocessing
def prepare_image(image):
    image = image.resize((256, 256))
    image = img_to_array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Main Route
@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    error_message = None

    if request.method == 'POST':
        image_file = request.files.get('image')
        image_url = request.form.get('image_url')
        image = None

        try:
            if image_file and image_file.filename != '':
                image = load_img(BytesIO(image_file.read()))

            elif image_url:
                response = requests.get(image_url, timeout=10)
                content_type = response.headers.get('Content-Type', '')

                if 'image' in content_type:
                    image = load_img(BytesIO(response.content))
                else:
                    error_message = "Provided URL does not contain an image file."

            if image:
                processed_image = prepare_image(image)
                probs = model.predict(processed_image)[0]
                sorted_indices = np.argsort(probs)[::-1]
                predictions = [
                    {'class_name': CLASS_NAMES[i], 'probability': float(probs[i])}
                    for i in sorted_indices
                ]
            elif not error_message:
                error_message = "Please upload an image file or provide a valid image URL."

        except Exception as e:
            error_message = f"An error occurred while processing the image: {str(e)}"

    return render_template_string(HTML_TEMPLATE, predictions=predictions, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1')
