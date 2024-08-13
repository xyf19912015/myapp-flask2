# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:25:30 2024

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:25:30 2024

@author: user
"""

from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve
import numpy as np
import requests
import io

app = Flask(__name__)

random_state = 42

# 加载数据
url = 'https://raw.githubusercontent.com/xyf19912015/myapp-flask2/main/JDM32SNE2XGB2.csv'
response = requests.get(url)
data = pd.read_csv(io.StringIO(response.content.decode('utf-8')), encoding='gbk')

print(data.columns.tolist())
data.columns = data.columns.str.strip()  # 移除列名中的前后空格

# 确定使用的特征
selected_features = [
    'Heliotrope Rash', 
    'Muscle Weakness', 
    'Interstitial Lung Disease',
    'Fever', 
    'LDH(U/L)', 
    'ESR(mm/h)', 
    'PLT(10^9/L)', 
    'AST(U/L)'
]

# 特征和标签
X = data[selected_features]
y = data['Cluster']

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 过采样处理不平衡
smote = SMOTE(sampling_strategy=1, random_state=random_state)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 训练集测试集分割
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=random_state)

# 构建模型
xgb_classifier = XGBClassifier(random_state=random_state)

# 网格搜索优化超参数
param_grid = {
    'n_estimators': [300],
    'learning_rate': [0.1],
    'max_depth': [3],
    'min_child_weight': [1],
    'gamma': [0],
    'subsample': [0.6],
    'colsample_bytree': [0.8],
    'reg_lambda': [1.0],
    'reg_alpha': [0.1]
}

grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_

# 计算最佳阈值
def calculate_best_threshold(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    return best_threshold

best_threshold = calculate_best_threshold(best_xgb, X_test, y_test)
print(best_threshold)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('input.html')

@app.route('/result', methods=['POST'])
def result():
    # 获取用户输入
    Heliotrope_Rash = 1 if request.form['Heliotrope_Rash'] == 'Yes' else 0
    Muscle_Weakness = 1 if request.form['Muscle_Weakness'] == 'Yes' else 0   
    Fever = 1 if request.form['Fever'] == 'Yes' else 0
    Interstitial_Lung_Disease = 1 if request.form['Interstitial_Lung_Disease'] == 'Yes' else 0
    AST = float(request.form['AST'])
    LDH = float(request.form['LDH'])
    ESR = float(request.form['ESR'])
    PLT = float(request.form['PLT'])

    # 创建输入数据
    user_input = pd.DataFrame([[Heliotrope_Rash, Muscle_Weakness, Fever, Interstitial_Lung_Disease, AST,LDH, ESR, PLT]], 
                              columns=selected_features)
    
    # 标准化用户输入
    user_input_scaled = scaler.transform(user_input)

    # 预测并分类
    y_proba = best_xgb.predict_proba(user_input_scaled)[:, 1]
    classification = 'Cluster 2(High risk)' if y_proba >= best_threshold else 'Cluster 1(Low risk)'

    return render_template('result.html', classification=classification)

if __name__ == '__main__':
    app.run(debug=True)
