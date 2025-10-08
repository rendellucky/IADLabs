import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Завантаження даних
file_path = './data_multivar_nb.txt'
data = np.loadtxt(file_path, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Створення та тренування класифікатора
classifier = GaussianNB()
classifier.fit(X, y)

# Прогнозування на тренувальних даних
y_pred = classifier.predict(X)

# Обчислення якості
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print(f"Accuracy of Naive Bayes classifier = {round(accuracy, 2)} %")

# Візуалізація результатів
visualize_classifier(classifier, X, y)

# Розбивка на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)

# Оцінка
acc_test = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print(f"Accuracy of the new classifier = {round(acc_test, 2)} %")
visualize_classifier(classifier_new, X_test, y_test)

# Перехресна перевірка
num_folds = 3
print("Accuracy:", round(100 * cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds).mean(), 2), "%")
print("Precision:", round(100 * cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds).mean(), 2), "%")
print("Recall:", round(100 * cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds).mean(), 2), "%")
print("F1:", round(100 * cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds).mean(), 2), "%")