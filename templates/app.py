from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # CORS 설정 추가

# 예측 모델 로드
model_regular = joblib.load('model_regular.pkl')
model_oversize = joblib.load('model_oversize.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    fit = request.form['fit']
    
    if fit == 'regular':
        prediction = model_regular.predict([[height, weight]])
    else:
        prediction = model_oversize.predict([[height, weight]])

    pants_length = round(prediction[0][0], 1)
    waist_size = round(prediction[0][1], 1)

    return jsonify({'pantsLength': pants_length, 'waistSize': waist_size})

if __name__ == '__main__':
    app.run(debug=True)
