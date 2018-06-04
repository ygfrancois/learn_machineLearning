from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
print(iris)
print(iris['target'].shape)
rf = RandomForestClassifier()
rf.fit(iris['data'][:150], iris['target'][:150])

instance = iris['data'][[50, 109]]
print(instance)
print(rf.predict(instance[[0]]))
print(rf.predict(instance[[1]]))
print(iris['target'][50], iris['target'][109])