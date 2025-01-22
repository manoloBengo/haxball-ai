# -------------------- Logistic Regression -------------------------------


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Crear el modelo con pesos personalizados
model_lr = LogisticRegression(class_weight={-1: 2, 0: 1, 1: 2}, random_state=random_state)

# Ajustar el modelo
model_lr.fit(X_train, y_train)

# Realizar predicciones
y_pred_lr = model_lr.predict(X_test)

# Evaluación: matriz de confusión
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=['Gol Blue', 'Gol Red', 'Ningún gol'], yticklabels=['Gol Blue', 'Gol Red', 'Ningún gol'])
plt.title("Matriz de Confusión - Logistic Regression")
plt.xlabel('Predicciones')
plt.ylabel('Valores reales')
plt.show()

# Reporte de clasificación
print("Reporte de clasificación - Logistic Regression:\n", classification_report(y_test, y_pred_lr, target_names=['Gol Blue', 'Gol Red', 'Ningún gol']))


from sklearn.model_selection import GridSearchCV

# Definir parámetros para GridSearch
param_grid_lr = {
    'C': [0.1, 1, 10],  # Parámetro de regularización
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 300]
}

# Configurar y realizar GridSearch
grid_search_lr = GridSearchCV(LogisticRegression(class_weight={-1: 2, 0: 1, 1: 2}, random_state=random_state), param_grid_lr, cv=5)
grid_search_lr.fit(X_train, y_train)

# Mostrar los mejores parámetros y resultados
print("Mejores parámetros - Logistic Regression:", grid_search_lr.best_params_)
print("Mejor puntuación - Logistic Regression:", grid_search_lr.best_score_)




# -------------------- Random forest -------------------------------


from sklearn.ensemble import RandomForestClassifier

# Crear el modelo con pesos personalizados
model_rf = RandomForestClassifier(class_weight={-1: 2, 0: 1, 1: 2}, random_state=random_state)

# Ajustar el modelo
model_rf.fit(X_train, y_train)

# Realizar predicciones
y_pred_rf = model_rf.predict(X_test)

# Evaluación: matriz de confusión
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Gol Blue', 'Gol Red', 'Ningún gol'], yticklabels=['Gol Blue', 'Gol Red', 'Ningún gol'])
plt.title("Matriz de Confusión - Random Forest")
plt.xlabel('Predicciones')
plt.ylabel('Valores reales')
plt.show()

# Reporte de clasificación
print("Reporte de clasificación - Random Forest:\n", classification_report(y_test, y_pred_rf, target_names=['Gol Blue', 'Gol Red', 'Ningún gol']))


param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Configurar y realizar GridSearch
grid_search_rf = GridSearchCV(RandomForestClassifier(class_weight={-1: 2, 0: 1, 1: 2}, random_state=random_state), param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)

# Mostrar los mejores parámetros y resultados
print("Mejores parámetros - Random Forest:", grid_search_rf.best_params_)
print("Mejor puntuación - Random Forest:", grid_search_rf.best_score_)



# -------------------- xg Boost -------------------------------


from xgboost import XGBClassifier

# Crear el modelo con parámetros personalizados
model_xgb = XGBClassifier(scale_pos_weight=2, random_state=random_state)

# Ajustar el modelo
model_xgb.fit(X_train, y_train)

# Realizar predicciones
y_pred_xgb = model_xgb.predict(X_test)

# Evaluación: matriz de confusión
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=['Gol Blue', 'Gol Red', 'Ningún gol'], yticklabels=['Gol Blue', 'Gol Red', 'Ningún gol'])
plt.title("Matriz de Confusión - XGBoost")
plt.xlabel('Predicciones')
plt.ylabel('Valores reales')
plt.show()

# Reporte de clasificación
print("Reporte de clasificación - XGBoost:\n", classification_report(y_test, y_pred_xgb, target_names=['Gol Blue', 'Gol Red', 'Ningún gol']))


param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, 2]
}

# Configurar y realizar GridSearch
grid_search_xgb = GridSearchCV(XGBClassifier(random_state=random_state), param_grid_xgb, cv=5)
grid_search_xgb.fit(X_train, y_train)

# Mostrar los mejores parámetros y resultados
print("Mejores parámetros - XGBoost:", grid_search_xgb.best_params_)
print("Mejor puntuación - XGBoost:", grid_search_xgb.best_score_)



# -------------------- SVM -------------------------------


from sklearn.svm import SVC

# Crear el modelo con pesos personalizados
model_svc = SVC(class_weight={-1: 2, 0: 1, 1: 2}, random_state=random_state)

# Ajustar el modelo
model_svc.fit(X_train, y_train)

# Realizar predicciones
y_pred_svc = model_svc.predict(X_test)

# Evaluación: matriz de confusión
cm_svc = confusion_matrix(y_test, y_pred_svc)
sns.heatmap(cm_svc, annot=True, fmt='d', cmap='Blues', xticklabels=['Gol Blue', 'Gol Red', 'Ningún gol'], yticklabels=['Gol Blue', 'Gol Red', 'Ningún gol'])
plt.title("Matriz de Confusión - Support Vector Machine")
plt.xlabel('Predicciones')
plt.ylabel('Valores reales')
plt.show()

# Reporte de clasificación
print("Reporte de clasificación - SVM:\n", classification_report(y_test, y_pred_svc, target_names=['Gol Blue', 'Gol Red', 'Ningún gol']))


param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3],
}

# Configurar y realizar GridSearch
grid_search_svc = GridSearchCV(SVC(class_weight={-1: 2, 0: 1, 1: 2}, random_state=random_state), param_grid_svc, cv=5)
grid_search_svc.fit(X_train, y_train)

# Mostrar los mejores parámetros y resultados
print("Mejores parámetros - SVM:", grid_search_svc.best_params_)
print("Mejor puntuación - SVM:", grid_search_svc.best_score_)
