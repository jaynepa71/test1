import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_and_save_model(csv_path, model_path):
    df = pd.read_csv(csv_path)
    X = df[['키(cm)', '몸무게(kg)']]
    y = df[['바지총장(cm)', '허리둘레(inch)']]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, model_path)

# 정핏 모델 학습 및 저장
train_and_save_model('regularsize (1).csv', 'model_regular.pkl')

# 오버핏 모델 학습 및 저장
train_and_save_model('overfit.csv', 'model_oversize.pkl')
