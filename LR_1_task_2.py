import numpy as np
from sklearn import preprocessing

# Варіант 8
input_data = np.array([
    [4.6, 9.9, -3.5],
    [-2.9, 4.1, 3.3],
    [-2.2, 8.8, -6.1],
    [3.9, 1.4, 2.2]
])

# 1. Бінаризація
data_binarized = preprocessing.Binarizer(threshold=2.2).transform(input_data)
print("Binarized data:\n", data_binarized)

# 2. Виключення середнього
print("\nBefore scaling:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)
print("\nAfter scaling:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

# 3. Масштабування (Min-Max)
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin-Max Scaled data:\n", data_scaled_minmax)

# 4. Нормалізація
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL1 Normalized data:\n", data_normalized_l1)
print("\nL2 Normalized data:\n", data_normalized_l2)
