from flask import Flask, request, jsonify
import csv
import os

app = Flask(__name__)

CSV_FILE = "person_Low_MWL.csv"

# --------- Utility to get next trial number ----------
def get_next_trial_number():
    if not os.path.exists(CSV_FILE):
        return 1
    with open(CSV_FILE, "r") as file:
        reader = csv.reader(file)
        headers = next(reader, [])
        return len(headers) + 1  # next trial number


# Initialize current trial number and header
CURRENT_TRIAL = get_next_trial_number()
TRIAL_HEADER = f"Trial {CURRENT_TRIAL}"

# If CSV doesn't exist, create it with first trial header
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([TRIAL_HEADER])


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask server is running!"})


@app.route("/data", methods=["POST"])
def receive_data():
    data = request.get_json()
    gsr_value = data.get("gsr_value", None)
    
    if gsr_value is None:
        return jsonify({"status": "error", "message": "Missing gsr_value"}), 400

    # Read existing CSV
    with open(CSV_FILE, "r") as file:
        reader = list(csv.reader(file))

    # Ensure first row is header
    if not reader:
        reader = [[TRIAL_HEADER]]
    elif len(reader[0]) < CURRENT_TRIAL:
        reader[0].append(TRIAL_HEADER)

    # Append new value to the column for current trial
    row_index = 1
    if len(reader) < row_index + 1:
        reader.append([])  # add new row if needed

    # Find first empty row in current trial column
    for i in range(1, len(reader)):
        while len(reader[i]) < CURRENT_TRIAL - 1:
            reader[i].append("")  # pad missing cells
        if len(reader[i]) < CURRENT_TRIAL or reader[i][CURRENT_TRIAL - 1] == "":
            reader[i].append(gsr_value)
            break
    else:
        # Add new row if all existing rows are filled
        new_row = [""] * (CURRENT_TRIAL - 1) + [gsr_value]
        reader.append(new_row)

    # Write back to CSV
    with open(CSV_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(reader)

    print(f"Saved Trial {CURRENT_TRIAL}: {gsr_value}")

    return jsonify({"status": "success", "trial": CURRENT_TRIAL, "saved_value": gsr_value})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
