import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utilities import visualize_classifier

# Завантаження даних
file_path = './data_multivar_nb.txt'
data = np.loadtxt(file_path, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбивка на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Створення SVM-класифікатора
classifier = SVC(kernel='rbf', gamma='auto', C=1.0)
classifier.fit(X_train, y_train)

# Прогнозування
y_pred = classifier.predict(X_test)

# Метрики
print("=== Support Vector Machine Classifier ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred, average='weighted') * 100, 2), "%")
print("Recall:", round(recall_score(y_test, y_pred, average='weighted') * 100, 2), "%")
print("F1:", round(f1_score(y_test, y_pred, average='weighted') * 100, 2), "%")

# Візуалізація
visualize_classifier(classifier, X_test, y_test)

# Перехресна перевірка
num_folds = 3
print("\nCross-validation results:")
print("Accuracy:", round(100 * cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds).mean(), 2), "%")
print("Precision:", round(100 * cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds).mean(), 2), "%")
print("Recall:", round(100 * cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds).mean(), 2), "%")
print("F1:", round(100 * cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds).mean(), 2), "%")