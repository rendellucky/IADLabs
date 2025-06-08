# LR_2_task_1.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Вхідний файл
input_file = 'income_data.txt'

# Завантаження даних
X = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Кодування ознак
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    try:
        X_encoded[:, i] = X[:, i].astype(float)
    except ValueError:
        encoder = preprocessing.LabelEncoder()
        X_encoded[:, i] = encoder.fit_transform(X[:, i])
        label_encoder.append(encoder)

# Розділення на X і y
X_features = X_encoded[:, :-1].astype(float)
y_target = X_encoded[:, -1].astype(int)

# Розділення на тренувальні та тестові набори
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=5)

# Створення та навчання класифікатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000))
classifier.fit(X_train, y_train)

# Прогнозування
y_test_pred = classifier.predict(X_test)

# Обчислення метрик
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# Прогноз для тестової точки
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

input_data_encoded = []
count = 0
for i, item in enumerate(input_data):
    try:
        input_data_encoded.append(float(item))
    except ValueError:
        encoder = label_encoder[count]
        input_data_encoded.append(encoder.transform([item])[0])
        count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)
predicted_class = classifier.predict(input_data_encoded)

# Декодування класу
predicted_label = label_encoder[-1].inverse_transform(predicted_class)
print("\nПрогноз для тестової точки:", predicted_label[0])
