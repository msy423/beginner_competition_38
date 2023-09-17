import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from loder.preprocess import preprocess_dataloder

pre = preprocess_dataloder()
X_train, X_test, y_train, y_test, scaler =pre()
column_means=X_train.mean()

# ハイパーパラメータの設定（必要に応じて調整）
n_estimators = 3000  # 決定木の数
learning_rate = 0.01  # 学習率
max_depth = 4  # 決定木の深さ

params = {
'objective': 'binary',  # 二値分類の場合
'boosting_type': 'gbdt',
'metric': 'binary_logloss',  # ログ尤度
'num_leaves': 31,  # 木の最大葉数
'learning_rate': 0.035,  # 学習率
'max_bin': 375,
'num_iterations':3000,
}

model = lgb.train(params, lgb.Dataset(X_train, label=y_train), num_boost_round=100) 

'''
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
# クロスバリデーションの各分割でのスコアを表示
for i, score in enumerate(cv_scores):
    print(f'Fold {i + 1} CV Score: {score}')

# 平均クロスバリデーションスコアを計算
mean_cv_score = cv_scores.mean()
print(f'Mean CV Score: {mean_cv_score}')
'''



'''
    #GBDT
    GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
            # ハイパーパラメータの設定（必要に応じて調整）
            n_estimators = 46  # 決定木の数
            learning_rate = 0.2  # 学習率
            max_depth = 3  # 決定木の深さ

    #LightGBM
        lgb.train(params, lgb.Dataset(X_train, label=y_train), num_boost_round=100) 
                params = {
                'objective': 'binary',  # 二値分類の場合
                'boosting_type': 'gbdt',
                'metric': 'binary_logloss',  # ログ尤度
                'num_leaves': 30,  # 木の最大葉数
                'learning_rate': 0.001,  # 学習率
                'max_bin': 450,
                'num_iterations':2500,
                }
            #threshold = 0.25
            #y_pred_binary = (y_pred > threshold).astype(int)

    #XGBoost
        #Best Parameters: {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 1000, 'subsample': 0.8}
            param_grid = {
            'n_estimators': [ 750, 1000, 1500 ,2500],
            'learning_rate': [0.2,0.05, 0.01, 0.005, 0.001],
            'max_depth': [3,4,5,6,7],
            'subsample': [0.7, 0.8, 0.9, 1.0],

            # 他のハイパーパラメータもここで指定
        }

        search_model = xgb.XGBClassifier(random_state=42)
        grid_search = GridSearchCV(search_model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f'Best Parameters: {best_params}')
        print(f'Best Score: {best_score}')

        model = xgb.XGBClassifier(**best_params, random_state=42)
'''

'''
'''
#model.fit(X_train, y_train)
y_pred = model.predict(X_test)

threshold = 0.3
y_pred = (y_pred > threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print(f'Confusion Matrix:\n{confusion}')

########################################################
print("\n---------------------------Submit set---------------------------\n")

test_data = pd.read_csv('data/test.csv')
test_data['Gender'] = test_data['Gender'].map({'Male': 0, 'Female': 1})
test_data['Gender'] = test_data['Gender'].map({'Male': 0, 'Female': 1})
test_data['Age_Tbill_ratio'] = test_data['Age']/test_data['T_Bil'] 
test_data['Age_Dbill_ratio'] = test_data['Age']/test_data['D_Bil'] 
test_data['ALTgpt_ASTgot_diff'] = test_data['ALT_GPT'] - test_data['AST_GOT'] 
test_data['Tbil_Dbil_diff'] = test_data['T_Bil'] - test_data['D_Bil'] 
test_data['ALTgpt_ASTgot_prod'] = test_data['ALT_GPT'] * test_data['AST_GOT'] 
test_data['Tbil_Dbil_prod'] = test_data['T_Bil'] * test_data['D_Bil']

means_by_disease_test = test_data.mean()
print("----------test means----------\n",means_by_disease_test)

test_data.fillna(column_means, inplace=True)

X_test_ = test_data  #test_data[['Gender', 'T_Bil', 'D_Bil', 'ALP', 'ALT_GPT', 'AST_GOT', 'TP', 'Alb']]
X_test_=scaler.transform(X_test_)
y_infe = model.predict(X_test_)
y_infe = (y_infe > threshold).astype(int)

id = [i for i in range(len(y_infe))]

output_data = pd.DataFrame(y_infe)
output_data.to_csv('predicted_results4.csv', header=False, index=True)

'''

'''


