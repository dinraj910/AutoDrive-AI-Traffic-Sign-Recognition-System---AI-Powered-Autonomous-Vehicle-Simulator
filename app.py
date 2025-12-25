from flask import Flask, render_template, request
import os
import tensorflow as tf
import numpy as np
from keras.preprocessing import image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

model = tf.keras.models.load_model("model/traffic_sign_cnn.h5")
IMG_SIZE = (64, 64)

CLASS_LABELS = [
    'cattle', 'give_way', 'narrow_bridge', 'narrow_road',
    'no_entry', 'no_left_turn', 'no_overtaking',
    'no_right_turn', 'speed_limit_90'
]

def predict(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    idx = np.argmax(preds)
    return CLASS_LABELS[idx], float(np.max(preds))

def decide_action(sign, confidence):
    speed = 60
    sign_type = "action"  # action, warning, danger

    if confidence < 0.6:
        return "Uncertain", "Low confidence prediction. Maintain speed.", speed, "warning"

    if sign == "no_entry":
        return "Stopped", "No entry. Vehicle stopped.", 0, "danger"

    if sign == "speed_limit_90":
        return "Running", "Speed set to 90 km/h.", 90, "action"

    if sign in ["narrow_road", "narrow_bridge", "cattle"]:
        return "Running", "Caution: " + sign.replace('_', ' ').title() + " ahead. Reducing speed.", int(speed * 0.6), "warning"

    if sign == "give_way":
        return "Running", "Give way to oncoming traffic.", int(speed * 0.7), "warning"

    if sign in ["no_left_turn", "no_right_turn", "no_overtaking"]:
        return "Running", sign.replace('_', ' ').title() + " - Restriction active.", speed, "action"

    return "Running", f"Notice: {sign.replace('_', ' ').title()}", speed, "action"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["image"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        sign, conf = predict(path)
        status, message, speed, sign_type = decide_action(sign, conf)

        result = {
            "sign": sign,
            "confidence": round(conf, 2),
            "status": status,
            "message": message,
            "speed": speed,
            "sign_type": sign_type
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
