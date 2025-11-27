from flask import Flask, request, jsonify
import csv
import os

app = Flask(__name__)

CSV_FILE = "personal_custom_Low_MWL.csv"

# ------ Initialize CSV with Trial Header ------
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Trial 1:0back", "Trial 6:0back"])  # EXACT header format


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask server is running!"})


@app.route("/data", methods=["POST"])
def receive_data():
    data = request.get_json()

    gsr_value = data.get("gsr_value", None)
    
    if gsr_value is None:
        return jsonify({"status": "error", "message": "Missing gsr_value"}), 400

    # Write to CSV with duplicate column (matching your sample)
    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([gsr_value, gsr_value])

    print(f"Saved: {gsr_value}")

    return jsonify({"status": "success", "saved_value": gsr_value})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
