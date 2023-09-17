import pandas as pd
import torch
import numpy as np

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.model_selection import train_test_split

from loder.preprocess import preprocess_dataloder

pre = preprocess_dataloder()
X_train, X_valid, y_train, y_valid, scaler = pre()


column_means=X_train.mean()
print(X_valid.values[0])



'''# 事前学習モデル
print(type(X_valid.values))
unsupervised_model = TabNetPretrainer(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type='entmax'
)

unsupervised_model.fit(
    X_train=X_train.values,
    eval_set=[X_valid.values],
    pretraining_ratio=0.8,
)

model = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10,
                      "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='sparsemax'
)

model.fit(
    X_train=X_train.values, y_train=y_train.values,
    eval_set=[(X_train.values, y_train.values), (X_valid.values, y_valid.values)],
    eval_name=['train', 'valid'],
    eval_metric=['auc'],
    #from_unsupervised=unsupervised_model # 事前学習したモデルの指定
)
#spreds = model.predict(X_test)'''
