import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski as scipy_minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def parse_embedding(embedding_str):
    embedding_str = embedding_str.strip()[1:-1]
    return np.array([float(x) for x in embedding_str.split(",")])


def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))


def euclidean_norm(a):
    return (sum(x * x for x in a)) ** 0.5


def mean_vector(x):
    return np.mean(x, axis=0)


def variance_vector(x):
    return np.var(x, axis=0)


def std_vector(x):
    return np.std(x, axis=0)


def interclass_distance(m1, m2):
    return np.linalg.norm(m1 - m2)


def feature_statistics(feature):
    return np.mean(feature), np.var(feature)


def minkowski_distance(a, b, p):
    return sum(abs(x - y) ** p for x, y in zip(a, b)) ** (1 / p)


def custom_knn(x_train, y_train, test_vec, k):
    distances = []
    for i in range(len(x_train)):
        d = euclidean_norm(x_train[i] - test_vec)
        distances.append((d, y_train[i]))
    distances.sort(key=lambda x: x[0])
    labels = [label for _, label in distances[:k]]
    return max(set(labels), key=labels.count)


def confusion_matrix_custom(y_true, y_pred):
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
    return tp, tn, fp, fn


def performance_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return accuracy, precision, recall, f1


if __name__ == "__main__":

    data = pd.read_csv("Coherence_bert_cls_embeddings.csv")

    X = np.array([parse_embedding(e) for e in data["embedding"]])
    y = data["label"].values

    classes = np.unique(y)
    X1 = X[y == classes[0]]
    X2 = X[y == classes[1]]

    v1 = X[0]
    v2 = X[1]

    print("Dot Product:", dot_product(v1, v2))
    print("NumPy Dot:", np.dot(v1, v2))
    print("Euclidean Norm:", euclidean_norm(v1))
    print("NumPy Norm:", np.linalg.norm(v1))

    m1 = mean_vector(X1)
    m2 = mean_vector(X2)
    print("Interclass Distance:", interclass_distance(m1, m2))

    feature = X[:, 0]
    mean_f, var_f = feature_statistics(feature)
    print("Feature Mean:", mean_f)
    print("Feature Variance:", var_f)

    plt.hist(feature, bins=15)
    plt.show()

    dist_list = []
    for p in range(1, 11):
        dist_list.append(minkowski_distance(v1, v2, p))

    plt.plot(range(1, 11), dist_list)
    plt.show()

    print("SciPy Minkowski:", scipy_minkowski(v1, v2, 3))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    print("kNN Accuracy:", knn.score(X_test, y_test))

    y_pred = knn.predict(X_test)
    print("Predictions:", y_pred[:5])

    custom_preds = []
    for vec in X_test:
        custom_preds.append(custom_knn(X_train, y_train, vec, 3))

    acc_list = []
    for k in range(1, 12):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc_list.append(model.score(X_test, y_test))

    plt.plot(range(1, 12), acc_list)
    plt.show()

    tp, tn, fp, fn = confusion_matrix_custom(y_test, y_pred)
    acc, prec, rec, f1 = performance_metrics(tp, tn, fp, fn)

    print("Confusion Matrix:", tp, tn, fp, fn)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
