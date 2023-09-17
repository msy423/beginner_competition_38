import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class preprocess_dataloder():
    def __init__(self):
        self.data=[]

    def __call__(self):
        data = pd.read_csv('data/train.csv')
        data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

        data_without_disease=data.drop('disease',axis=1).copy()
        column_means = data.mean()
        means_by_disease = data.groupby('disease').mean()

        #print("----------train means----------\n",means_by_disease)

        z_scores = (data - data.mean()) / data.std()

        # Zスコアがしきい値を超える行を抽出
        threshold = 3
        outliers = data[(z_scores > threshold).any(axis=1)]
        # 外れ値を含む行を削除
        cleaned_data = data.drop(outliers.index)

        # インデックスをリセット（オプション）
        cleaned_data.reset_index(drop=True, inplace=True)


        #age, alp, TP Alb,AG_ratioはあまり平均に差がない

        X = cleaned_data.drop('disease', axis=1)  #[['T_Bil', 'D_Bil', 'ALP', 'ALT_GPT', 'AST_GOT', 'TP', 'Alb']]
        y = cleaned_data['disease']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train))
        X_train.columns=X.columns #['Gender','T_Bil', 'D_Bil', 'ALP', 'ALT_GPT', 'AST_GOT', 'TP', 'Alb']


        X_test=pd.DataFrame(scaler.transform(X_test))
        X_test.columns=X.columns

        self.data=[ X_train, X_test, y_train, y_test, scaler ]

        return self.data
        
if __name__ == "__main__":
    pre = preprocess_dataloder()
    X_train, X_test, y_train, y_test =pre()
    print(X_train)