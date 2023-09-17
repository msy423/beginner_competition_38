import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import seaborn as sns
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('data/train.csv')
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

column_means = data.mean()
means_by_disease = data.groupby('disease').mean()

z_scores = (data - data.mean()) / data.std()

# Zスコアがしきい値を超える行を抽出
threshold = 3
outliers = data[(z_scores > threshold).any(axis=1)]
# 外れ値を含む行を削除
cleaned_data = data.drop(outliers.index)

# インデックスをリセット（オプション）
cleaned_data.reset_index(drop=True, inplace=True)


print(len(outliers))
print(cleaned_data)




#age, alp, TP Alb,AG_ratioはあまり平均に差がない

X = data[['T_Bil', 'D_Bil', 'ALP', 'ALT_GPT', 'AST_GOT', 'TP', 'Alb']]
y = data['disease']
from sklearn.ensemble import IsolationForest
# Isolation Forestを使用して外れ値を検出



#scaler = StandardScaler()
#X_scaled = pd.concat([data['Gender'], pd.DataFrame(scaler.fit_transform(X))], axis=1)
#X_scaled.columns=['Gender','T_Bil', 'D_Bil', 'ALP', 'ALT_GPT', 'AST_GOT', 'TP', 'Alb']
#print(X)
#print(X_scaled)

'''
sns.set(style="whitegrid")
plt.figure(figsize=(16, 12))

# 項目リストを作成
columns = ['Age', 'T_Bil', 'D_Bil', 'ALP', 'ALT_GPT', 'AST_GOT', 'TP', 'Alb', 'AG_ratio']

# サブプロットのレイアウトを設定
num_rows = 3
num_cols = 3

for i, column in enumerate(columns, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.histplot(data=data, x=column, hue='disease', palette='colorblind', element='step', common_norm=False)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column} (colored by disease)')
    plt.legend(title='Disease', labels=['No Disease', 'Disease'])

plt.tight_layout()
plt.show()
'''