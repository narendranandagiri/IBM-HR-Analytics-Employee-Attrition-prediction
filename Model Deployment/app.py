from flask import Flask, request, jsonify, render_template
import pickle
from predict import predict_response

app = Flask(__name__)

# Loading the model and scaler
model = pickle.load(open('../Artifacts/model.pkl', 'rb'))
scaling = pickle.load(open('../Artifacts/scaling.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        if not input_data:
            raise ValueError("No input data provided")
        
        result = predict_response(input_data)
        
        return jsonify({
            'message': result
        })
    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
