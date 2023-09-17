cv_scores = cross_val_score(model, X, y, cv=5)
# クロスバリデーションの各分割でのスコアを表示
for i, score in enumerate(cv_scores):
    print(f'Fold {i + 1} CV Score: {score}')

# 平均クロスバリデーションスコアを計算
mean_cv_score = cv_scores.mean()
print(f'Mean CV Score: {mean_cv_score}')