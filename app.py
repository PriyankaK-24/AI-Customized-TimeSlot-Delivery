from flask import Flask, render_template, request
import joblib
import pandas as pd
import datetime
import pyttsx3
import csv
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "slot_model.joblib"   # change to .pkl if renamed
if os.path.exists(MODEL_PATH):
    clf = joblib.load(MODEL_PATH)
else:
    clf = None
    print("⚠️  slot_model.joblib not found — predictions will be random")

SLOTS = ["9-11 AM", "11-1 PM", "1-3 PM", "3-5 PM", "5-7 PM"]

# ============= Routes =============

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    order_id = request.form["order_id"]
    name = request.form["name"]
    address = request.form["address"]
    guide = request.form["guide"]

    # --- simple fake input for model ---
    sample_data = pd.DataFrame([[len(name), len(address), len(guide)]],
                               columns=["name_len", "addr_len", "guide_len"])

    if clf is not None:
        pred = clf.predict(sample_data)[0]
        slot = SLOTS[int(pred) % len(SLOTS)]
        confidence = 0.85
    else:
        slot = SLOTS[0]
        confidence = 0.5

    return render_template("predict.html",
                           order_id=order_id,
                           name=name,
                           address=address,
                           guide=guide,
                           slot=slot,
                           confidence=confidence)

@app.route("/confirm", methods=["POST"])
def confirm():
    order_id = request.form["order_id"]
    name = request.form["name"]
    address = request.form["address"]
    slot = request.form["slot"]
    guide = request.form["guide"]
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    # --- save to CSV ---
    file_exists = os.path.isfile("confirmed.csv")
    with open("confirmed.csv", mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Order ID", "Name", "Address", "Slot", "Guide", "Timestamp"])
        writer.writerow([order_id, name, address, slot, guide, timestamp])
    print(f"✅ Booking saved for {order_id}")

    # --- voice assistant ---
    try:
        engine = pyttsx3.init()
        engine.say(f"Booking confirmed for {name}. Delivery slot is {slot}. "
                   f"Instructions for delivery: {guide}")
        engine.runAndWait()
    except Exception as e:
        print("Voice assistant error:", e)

    return render_template("confirmed.html",
                           order_id=order_id,
                           name=name,
                           address=address,
                           slot=slot,
                           guide=guide,
                           timestamp=timestamp)

@app.route("/partner")
def partner():
    if os.path.exists("confirmed.csv"):
        df = pd.read_csv("confirmed.csv")
        data = df.to_dict(orient="records")
    else:
        data = []
    return render_template("partner.html", bookings=data)

@app.route("/test")
def test():
    return "✅ Flask is running fine!"

if __name__ == "__main__":
    print("Starting:", __file__)
    app.run(host="127.0.0.1", port=5001, debug=True)
