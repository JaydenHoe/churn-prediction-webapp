from flask import Flask, request, render_template, jsonify
from model import Model

# Churn classification threshold chosen from notebook
# (best F1 on validation set at ~0.27)
THRESHOLD = 0.27

app = Flask(__name__)
model = Model()

#Method 1: Via HTML Form
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

#Method 2: Via POST API
@app.route('/api/predict', methods=['POST'])
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        # fallback: allow form posts for quick manual testing
        payload = request.form.to_dict()

    try:
        if isinstance(payload, list):
            results = model.predict_batch(payload)
            return jsonify({"success": True, "results": results}), 200

        result = model.predict_one(payload)
        prob = result.get("churn_probability") or result.get("probability")
        will_churn = None
        if prob is not None:
            will_churn = prob >= THRESHOLD

        response = {"success": True, **result}
        if will_churn is not None:
            response["will_churn"] = will_churn
        return jsonify(response), 200
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception:
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route('/api/features', methods=['GET'])
def features():
    return jsonify({"success": True, **model.schema()}), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
    