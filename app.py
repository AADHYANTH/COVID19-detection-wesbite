from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your Keras model
model = tf.keras.models.load_model("covid_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image = request.files['image']
    image_np = tf.image.decode_image(image.read(), channels=3)
    image_np = tf.image.resize(image_np, (224, 224)) / 255.0

    # Perform model prediction
    prediction = model.predict(np.expand_dims(image_np, axis=0))
    class_names = ["covid", "normal", "virus"]
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    messages = {
        "covid": "Based on the analysis, The X-ray shows signs of a COVID-19 infection.",
        "normal": "Great news! The X-ray appears to be normal with no signs of infection.",
        "virus": "The X-ray indicates a viral infection. Further evaluation may be needed."
    }
    return render_template('index.html', prediction=predicted_class, message=messages[predicted_class])


if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load your Keras model
model = tf.keras.models.load_model("covid_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image = request.files['image']
    image_np = tf.image.decode_image(image.read(), channels=3)
    image_np = tf.image.resize(image_np, (224, 224)) / 255.0

    # Perform model prediction
    prediction = model.predict(np.expand_dims(image_np, axis=0))
    class_names = ["covid", "normal", "virus"]
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    messages = {
        "covid": "Based on the analysis, The X-ray shows signs of a COVID-19 infection.",
        "normal": "Great news! The X-ray appears to be normal with no signs of infection.",
        "virus": "The X-ray indicates a viral infection. Further evaluation may be needed."
    }
    return render_template('index.html', prediction=predicted_class, message=messages[predicted_class])


if __name__ == '__main__':
    app.run(debug=True)
