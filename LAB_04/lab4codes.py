import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler

def load_dataset(file_path):
    data = pd.read_csv(file_path)
    return data

def split_features_target(data, target_column):
    X = data.drop(columns=[target_column])
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0)
    y = data[target_column]
    return X, y

def train_knn_classifier(X_train, y_train, k_value=5):
    model = KNeighborsClassifier(n_neighbors=k_value)
    model.fit(X_train, y_train)
    return model

def evaluate_classification(model, X, y):
    predictions = model.predict(X)
    cm = confusion_matrix(y, predictions)
    precision = precision_score(y, predictions, average='weighted', zero_division=0)
    recall = recall_score(y, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y, predictions, average='weighted', zero_division=0)
    return cm, precision, recall, f1

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

def generate_training_points():
    np.random.seed(42)
    X = np.random.randint(1, 11, size=(20, 2))
    y = np.random.randint(0, 2, size=20)
    return X, y

def plot_training_points(X, y):
    colors = ['blue' if label == 0 else 'red' for label in y]
    plt.scatter(X[:,0], X[:,1], c=colors)
    plt.title("Training Data Scatter Plot")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.show()

def generate_test_grid():
    x_range = np.arange(0, 10, 0.1)
    y_range = np.arange(0, 10, 0.1)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    return grid_points

def classify_and_plot(model, grid_points):
    predictions = model.predict(grid_points)
    colors = ['blue' if label == 0 else 'red' for label in predictions]
    plt.scatter(grid_points[:,0], grid_points[:,1], c=colors, s=1)
    plt.title("Decision Boundary Visualization")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.show()

def test_multiple_k_values(X_train, y_train, grid_points, k_values):
    for k in k_values:
        model = train_knn_classifier(X_train, y_train, k)
        classify_and_plot(model, grid_points)

def project_feature_plot(data, feature1, feature2, target_column):
    colors = ['blue' if label == 0 else 'red' for label in data[target_column]]
    plt.scatter(data[feature1], data[feature2], c=colors)
    plt.title("Project Data Scatter Plot")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()

def tune_k_value(X_train, y_train):
    param_grid = {'n_neighbors': list(range(1, 21))}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_

file_path = r"C:\Users\Divya Rangumudri\Downloads\SEM-4\ML\LAB4\Coherence_bert_cls_embeddings.csv"
target_column = "label"

data = load_dataset(file_path)
X, y = split_features_target(data, target_column)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = train_knn_classifier(X_train, y_train)

train_cm, train_precision, train_recall, train_f1 = evaluate_classification(model, X_train, y_train)
test_cm, test_precision, test_recall, test_f1 = evaluate_classification(model, X_test, y_test)

print("training results")
print(train_cm)
print(train_precision, train_recall, train_f1)

print("testing_result")
print(test_cm)
print(test_precision, test_recall, test_f1)

train_points, train_labels = generate_training_points()
plot_training_points(train_points, train_labels)

grid_points = generate_test_grid()
model_random = train_knn_classifier(train_points, train_labels, k_value=3)
classify_and_plot(model_random, grid_points)

test_multiple_k_values(train_points, train_labels, grid_points, [1,3,5,7])

feature1 = data.select_dtypes(include=[np.number]).columns[0]
feature2 = data.select_dtypes(include=[np.number]).columns[1]
project_feature_plot(data, feature1, feature2, target_column)

best_k, best_score = tune_k_value(X_train, y_train)
print("Best k:", best_k)
print("Best CV Score:", best_score)
