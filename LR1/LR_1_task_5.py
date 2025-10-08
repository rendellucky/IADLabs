import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score,
    precision_score, f1_score, roc_curve, roc_auc_score
)

data = {
    'actual_label': [1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    'predicted_RF': [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0],
    'model_RF': [0.9, 0.8, 0.6, 0.3, 0.4, 0.7, 0.1, 0.9, 0.2, 0.6, 0.9, 0.2],
    'model_LR': [0.8, 0.1, 0.2, 0.3, 0.1, 0.7, 0.1, 0.8, 0.2, 0.2, 0.9, 0.1]
}
df = pd.DataFrame(data)

# Перетворення ймовірностей у класи
df['predicted_RF'] = (df.model_RF >= 0.5).astype(int)
df['predicted_LR'] = (df.model_LR >= 0.5).astype(int)

# --- Власні функції (приклад для студента Konev) ---

def konev_find_TP(y_true, y_pred): return sum((y_true == 1) & (y_pred == 1))
def konev_find_FN(y_true, y_pred): return sum((y_true == 1) & (y_pred == 0))
def konev_find_FP(y_true, y_pred): return sum((y_true == 0) & (y_pred == 1))
def konev_find_TN(y_true, y_pred): return sum((y_true == 0) & (y_pred == 0))

def konev_find_conf_matrix_values(y_true, y_pred):
    TP = konev_find_TP(y_true, y_pred)
    FN = konev_find_FN(y_true, y_pred)
    FP = konev_find_FP(y_true, y_pred)
    TN = konev_find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def konev_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = konev_find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

def konev_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = konev_find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def konev_recall_score(y_true, y_pred):
    TP, FN, FP, TN = konev_find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)

def konev_precision_score(y_true, y_pred):
    TP, FN, FP, TN = konev_find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)

def konev_f1_score(y_true, y_pred):
    recall = konev_recall_score(y_true, y_pred)
    precision = konev_precision_score(y_true, y_pred)
    return 2 * precision * recall / (precision + recall)

# --- Перевірка відповідності sklearn ---
y_true = df.actual_label.values
y_pred_RF = df.predicted_RF.values
y_pred_LR = df.predicted_LR.values

assert np.array_equal(konev_confusion_matrix(y_true, y_pred_RF),
                      confusion_matrix(y_true, y_pred_RF))
assert abs(konev_accuracy_score(y_true, y_pred_RF)
           - accuracy_score(y_true, y_pred_RF)) < 1e-9

# --- Виведення результатів ---
print("=== Random Forest ===")
print("Accuracy:", konev_accuracy_score(y_true, y_pred_RF))
print("Recall:", konev_recall_score(y_true, y_pred_RF))
print("Precision:", konev_precision_score(y_true, y_pred_RF))
print("F1:", konev_f1_score(y_true, y_pred_RF))

print("\n=== Logistic Regression ===")
print("Accuracy:", konev_accuracy_score(y_true, y_pred_LR))
print("Recall:", konev_recall_score(y_true, y_pred_LR))
print("Precision:", konev_precision_score(y_true, y_pred_LR))
print("F1:", konev_f1_score(y_true, y_pred_LR))

# --- ROC-криві ---
fpr_RF, tpr_RF, _ = roc_curve(y_true, df.model_RF.values)
fpr_LR, tpr_LR, _ = roc_curve(y_true, df.model_LR.values)

auc_RF = roc_auc_score(y_true, df.model_RF.values)
auc_LR = roc_auc_score(y_true, df.model_LR.values)

plt.plot(fpr_RF, tpr_RF, 'r-', label=f'RF AUC: {auc_RF:.3f}')
plt.plot(fpr_LR, tpr_LR, 'b-', label=f'LR AUC: {auc_LR:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curves')
plt.show()

print(f"\nAUC RF = {auc_RF:.3f}")
print(f"AUC LR = {auc_LR:.3f}")