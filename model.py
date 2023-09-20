import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
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
'learning_rate': 0.01,  # 学習率
'max_bin': 5000,
'num_iterations':10000,
'subsmaple': 0.8,
'silent':True
}




'''


'''

#['Age', 'Gender', 'T_Bil', 'D_Bil', 'ALP', 'ALT_GPT', 'AST_GOT', 'TP',
#       'Alb', 'AG_ratio', 'TP_Tbil_ratio', 'ALTGPT_TP_ratio', 'Age_Tbil_ratio',
#       'Age_Dbil_ratio', 'Age_ALP_ratio', 'Age_ALTGPT_ratio',
#       'Age_ASTGOT_ratio', 'Age_TP_ratio', 'Age_ALb_ratio', 'Age_AG_ratio',
#       'ALTgpt_ASTgot_diff', 'Tbil_Dbil_diff', 'ALTgpt_ASTgot_prod',
#       'Tbil_Dbil_prod'],


#print(X_train.columns)

gbm_col =[ 'T_Bil', 'D_Bil', 'ALP', 'ALT_GPT','AG_ratio', 'TP_Tbil_ratio',
          'ALTGPT_TP_ratio', 'Age_Dbil_ratio', 'Age_ALP_ratio', 'Age_ALTGPT_ratio', 
          'Age_ASTGOT_ratio', 'Age_TP_ratio', 'Age_ALb_ratio', 'Age_AG_ratio', 'ALTgpt_ASTgot_diff','ALTgpt_ASTgot_prod' ]

logi_model = LogisticRegression() 
svm_model = SVC(kernel='rbf', C=1)
gbm = lgb.train(params, lgb.Dataset(X_train[gbm_col], label=y_train), num_boost_round=100) 


logi_col=['T_Bil', 'D_Bil', 'ALT_GPT', 'Alb',  'TP_Tbil_ratio', 'Age_TP_ratio', 'Age_AG_ratio', 'ALTgpt_ASTgot_prod']
logi_model.fit(X_train[logi_col],y_train)
y_pred_logi=logi_model.predict(X_test[logi_col])

svm_col=[ 'AST_GOT', 'TP_Tbil_ratio', 'ALTGPT_TP_ratio', 'Age_ALP_ratio', 'Age_ALTGPT_ratio', 'Age_ASTGOT_ratio', 'Age_TP_ratio', 'Age_AG_ratio', 'Tbil_Dbil_diff']
svm_model.fit(X_train[svm_col], y_train)
y_pred_svm = svm_model.predict(X_test[svm_col])

y_pred_gbm = gbm.predict(X_test[gbm_col])
threshold = 0.5
y_pred_gbm = (y_pred_gbm > threshold).astype(int)

y_pred = ((y_pred_logi+y_pred_gbm+y_pred_svm)/3 > 0.5).astype(int)

y_preds=pd.DataFrame({'gt':y_test, 'logistic':y_pred_logi, 'SVM':y_pred_svm, 'LightGBM':y_pred_gbm ,'mean':y_pred})


y_preds.to_csv('ensamble_experiment_3.csv', header=False, index=True)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print(f'Confusion Matrix:\n{confusion}')




########################################################

print("\n---------------------------Submit set---------------------------\n")

data = pd.read_csv('data/test.csv')
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
column_med = #median 
data.fillna(column_med, inplace=True)

data['TP_Tbil_ratio'] = data['TP']/data['T_Bil'] 
data['ALTGPT_TP_ratio'] = data['ALT_GPT']/data['TP']


data['Age_Tbil_ratio'] = data['Age']/data['T_Bil'] 
data['Age_Dbil_ratio'] = data['Age']/data['D_Bil']
data['Age_ALP_ratio'] = data['Age']/data['ALP']
data['Age_ALTGPT_ratio'] = data['Age']/data['ALT_GPT']
data['Age_ASTGOT_ratio'] = data['Age']/data['AST_GOT']
data['Age_TP_ratio'] = data['Age']/data['TP']
data['Age_ALb_ratio'] = data['Age']/data['Alb']
data['Age_AG_ratio'] = data['Age']/data['AG_ratio']

data['ALTgpt_ASTgot_diff'] = data['ALT_GPT'] - data['AST_GOT'] 
data['Tbil_Dbil_diff'] = data['T_Bil'] - data['D_Bil'] 
data['ALTgpt_ASTgot_prod'] = data['ALT_GPT'] * data['AST_GOT'] 
data['Tbil_Dbil_prod'] = data['T_Bil'] * data['D_Bil'] 

means_by_disease_test = test_data.mean()
print("----------test means----------\n",means_by_disease_test)



X_test_ = test_data  #test_data[['Gender', 'T_Bil', 'D_Bil', 'ALP', 'ALT_GPT', 'AST_GOT', 'TP', 'Alb']]
X_test_=scaler.transform(X_test_)
y_infe = model.predict(X_test_)
y_infe = (y_infe > threshold).astype(int)

id = [i for i in range(len(y_infe))]

output_data = pd.DataFrame(y_infe)
output_data.to_csv('predicted_results4.csv', header=False, index=True)

'''

'''


######################################################
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
