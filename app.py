import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU (Force CPU usage)
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# Load the trained AI model
model = tf.keras.models.load_model("soil_moisture_lstm.h5", compile=False)

# Recompile model with correct loss function
model.compile(loss="mean_squared_error", optimizer="adam")

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to AgriNexus AI API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data["soil_moisture"]).reshape(1, 10, 1)

        prediction = model.predict(input_data)
        return jsonify({"predicted_moisture": float(prediction[0][0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)  # Debug mode disabled
