import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# CSV 파일에서 데이터 로드하는 함수
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df[['키(cm)', '몸무게(kg)']]  # 특성 데이터
    X.columns = ['키', '몸무게']  # 특성 이름 지정
    y = df[['바지총장(cm)', '허리둘레(inch)']]  # 타겟 데이터
    return X, y


# 모델 학습 함수
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# 입력에 대한 예측을 수행하는 함수
def predict_size(model, height, weight):
    prediction = model.predict([[height, weight]])
    pants_length = prediction[0][0]
    waist_size = prediction[0][1]
    return pants_length, waist_size

# CSV 파일 경로
regular_csv_path = 'regularsize (1).csv'
oversize_csv_path = 'overfit.csv'

# 데이터 로드
X_regular, y_regular = load_data(regular_csv_path)
X_oversize, y_oversize = load_data(oversize_csv_path)

# 모델 학습
model_regular = train_model(X_regular, y_regular)
model_oversize = train_model(X_oversize, y_oversize)

# 예측 테스트
height = 170
weight = 65
pants_length_regular, waist_size_regular = predict_size(model_regular, height, weight)
pants_length_oversize, waist_size_oversize = predict_size(model_oversize, height, weight)

print("정핏 바지 총장: {:.1f} cm".format(pants_length_regular))
print("정핏 허리 둘레: {:.1f} inch".format(waist_size_regular))
print("오버핏 바지 총장: {:.1f} cm".format(pants_length_oversize))
print("오버핏 허리 둘레: {:.1f} inch".format(waist_size_oversize))
