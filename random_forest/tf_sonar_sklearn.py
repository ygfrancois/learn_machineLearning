import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn import metrics
import matplotlib.pylab as plt


train = pd.read_csv('sonar.all-data.csv')
target='R'  # Disbursed的值就是二元分类的输出
print(train['R'].value_counts())

x_columns = [x for x in train.columns if x != target]
X = train[x_columns]
y = train['R']
y = y.replace(to_replace={'R': 0, 'M': 1})

rf0 = RandomForestClassifier(oob_score=True, random_state=10)  # random_state:随机数种子，初始化随机数
rf0.fit(X, y)
print(rf0.oob_score_)  # 袋外预测正确分数，可以表现模型的泛化程度
y_predict = rf0.predict(X)

pred = [y[i] == y_predict[i] for i in range(0, len(y))]
accuracy = pred.count(True) / len(pred)
print(accuracy)

y_predprob = rf0.predict_proba(X)[:, 1]
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

param_test1 = {'n_estimators': range(50, 71, 2)}
gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
                                                         min_samples_leaf=20,
                                                         max_depth=8,
                                                         max_features='sqrt',
                                                         random_state=10),
                        param_grid=param_test1, scoring='roc_auc', cv=5)

gsearch1.fit(X, y)
print(gsearch1.cv_results_.get('mean_test_score'))
print(gsearch1.best_params_, gsearch1.best_score_)

# 这样我们得到了最佳的弱学习器迭代次数，接着我们对决策树最大深度max_depth和
# 内部节点再划分所需最小样本数min_samples_split进行网格搜索。
param_test2 = {'max_depth': range(1, 5, 1), 'min_samples_split': range(2, 10, 1)}
gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=56, min_samples_leaf=20, max_features='sqrt',
                                                         oob_score=True, random_state=10),
                        param_grid=param_test2, scoring='roc_auc', cv=5)
gsearch2.fit(X, y)
print(gsearch2.cv_results_.get('mean_test_score'))
print(gsearch2.best_params_, gsearch2.best_score_)

rf1 = RandomForestClassifier(n_estimators=56, max_depth=3, min_samples_split=2, min_samples_leaf=2, max_features='sqrt',
                             oob_score=True, random_state=10)
rf1.fit(X, y)
print(rf1.oob_score_)

# 可见此时我们的袋外分数有一定的提高。也就是时候模型的泛化能力增强了。对于内部节点再划分所需最小样本数min_samples_split，我们暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。下面我们再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参。

# 再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
param_test3 = {'min_samples_split': range(2, 10, 1), 'min_samples_leaf': range(5, 20, 1)}
gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=56, max_features='sqrt', oob_score=True,
                                                         random_state=10, max_depth=3),
                        param_grid=param_test3, scoring='roc_auc', cv=5)
gsearch3.fit(X, y)
print(gsearch3.cv_results_.get('mean_test_score'))
print(gsearch3.best_params_, gsearch3.best_score_)

param_test4 = {'max_features': range(5, 9, 1)}
gsearch4 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=56, min_samples_split=2, min_samples_leaf=10,
                                                         oob_score=True, random_state=10, max_depth=3),
                        param_grid=param_test4, scoring='roc_auc', cv=5)
gsearch4.fit(X, y)
print(gsearch4.cv_results_.get('mean_test_score'))
print(gsearch4.best_params_, gsearch4.best_score_)

rf2 = RandomForestClassifier(n_estimators=56, min_samples_leaf=10, min_samples_split=2,
                             oob_score=True, random_state=10, max_features=7)
rf2.fit(X, y)
print(rf2.oob_score_)