from flask import Flask, jsonify, request

app = Flask(__name__)

# Test route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask server is running!"})

# Example POST route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Just echo the input for now
    return jsonify({
        "received_data": data,
        "status": "success"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
