from random import seed
from random import randrange
from csv import reader
from math import sqrt


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    # 将所有的分类标签用字典形式表达
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
       lookup[value] = i
    for row in dataset:
       row[column] = lookup[row[column]]
    return lookup


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
   dataset_split = list()
   dataset_copy = list(dataset)
   fold_size = len(dataset) / n_folds
   for i in range(n_folds):
       fold = list()
       while len(fold) < fold_size and len(dataset_copy) > 0:
           index = randrange(len(dataset_copy))  # randrange 返回参数范围内的一个随机数
           fold.append(dataset_copy.pop(index))
       dataset_split.append(fold)
   return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
   correct = 0
   for i in range(len(actual)):
       if actual[i] == predicted[i]:
           correct += 1
   return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)  # 划分训练集和测试集
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)  # 用k中的一个数据集测试，其余的用来训练
        train_set = sum(train_set, [])  # 改变一下矩阵纬度
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)

    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    # group 是被分裂后的数据集list
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) /float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


def get_split(dataset, n_features):
    # dataset是train样本
    class_values = list(set(row[-1] for row in dataset))  # 获取分类值
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    while len(features) < n_features:
        # 随即获取分裂特征，直到个数达到n_features
        index = randrange(len(dataset[0]) - 1)
        if index not in features:
            features.append(index)
    for index in features:
        for row in dataset:  # 遍历每一种特征在每个样本中的值，找出gini值最小的二分类
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups

    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
   outcomes = [row[-1] for row in group]
   return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])

    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
       node['left'], node['right'] = to_terminal(left), to_terminal(right)
       return
    # process left child
    if len(left) <= min_size:
       node['left'] = to_terminal(left)
    else:
       node['left'] = get_split(left, n_features)
       split(node['left'], max_depth, min_size, n_features, depth+1)
    # process right child
    if len(right) <= min_size:
       node['right'] = to_terminal(right)
    else:
       node['right'] = get_split(right, n_features)
       split(node['right'], max_depth, min_size, n_features, depth+1)


def build_tree(dataset, max_depth, min_size, n_features):
    root = get_split(dataset, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root


def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']

    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)  # 如果sample_size不为1.0， 训练集样本的选取是原始数据集的子集
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return predictions


if __name__=='__main__':
    # Test the random forest algorithm
    seed(1)
    # load and prepare data
    filename = 'sonar.all-data.csv'
    dataset = load_csv(filename)
    # convert string attributes to float
    for i in range(0, len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)

    n_folds = 5  # 交叉评估
    max_depth = 10  # 每棵树最大深度为10
    min_size = 1  # 每个节点最小训练行数为1
    sample_size = 1.0  # 创建训练集样本的大小与原始数据集相同
    n_features = int(sqrt(len(dataset[0]) -1))
    for n_trees in [10]:
        scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth,
                                    min_size, sample_size, n_trees, n_features)
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
