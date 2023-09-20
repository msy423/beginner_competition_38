import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class preprocess_dataloder():
    def __init__(self):
        self.data=[]

    def __call__(self):
        data = pd.read_csv('data/train.csv')
        data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
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



        #data_without_disease=data.drop('disease',axis=1).copy()
        #column_means = data.mean()
        #means_by_disease = data.groupby('disease').mean()

        #print("----------train means----------\n",means_by_disease)

        
        z_scores = (data - data.mean()) / data.std()

        # Zスコアがしきい値を超える行を抽出
        threshold = 3
        outliers = data[(z_scores > threshold).any(axis=1)]
        # 外れ値を含む行を削除
        cleaned_data = data.drop(outliers.index)

        # インデックスをリセット（オプション）
        cleaned_data.reset_index(drop=True, inplace=True)
        '''

        threshold = 3.0
        for column in ['Age', 'T_Bil', 'D_Bil', 'ALP', 'ALT_GPT', 'AST_GOT', 'TP', 'Alb', 'AG_ratio']:
            z_scores = (data[column] - data[column].mean()) / data[column].std()
            outliers = z_scores.abs() > threshold
            data.loc[outliers, column] = data[column].mean()

        cleaned_data= data
        '''
        #age, alp, TP Alb,AG_ratioはあまり平均に差がない

        X = cleaned_data.drop(['disease'], axis=1)  #[['T_Bil', 'D_Bil', 'ALP', 'ALT_GPT', 'AST_GOT', 'TP', 'Alb']]
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