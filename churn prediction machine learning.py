import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

df=pd.read_csv("https://raw.githubusercontent.com/nikhilsthorat03/Telco-Customer-Churn/refs/heads/main/telco.csv")
print(df.shape)
print(df.head(40))

print("\nChurn distribution:")
print(df['Churn'].value_counts())
print(df["Churn"].value_counts(normalize=True))

print("\nMissing values:")
print(df.isnull().sum())

print("\n Data Types:")
print(df.dtypes)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df = df.dropna()

df = df.drop('customerID', axis=1)

print(f"\nDataset shape after cleaning: {df.shape}")

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Churn')

print(f"\nCategorical columns to encode: {categorical_cols}")

label_encoders = {}
for col in categorical_cols:
    LE = LabelEncoder()
    df[col] = LE.fit_transform(df[col])
    label_encoders[col] = LE

x=df.drop("Churn", axis=1)
y=df["Churn"]

print(f"\nFeatures shape: {x.shape}")
print(f"Target shape: {y.shape}")
print("\nFeatures being used:")
print(x.columns.tolist())

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=42)

print(f"\nTraining set: {x_train.shape}")
print(f"Test set: {x_test.shape}")

svm_pipeline=Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42))
])

knn_pipeline=Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

rf=RandomForestClassifier(random_state=42, n_estimators=100)

cv_strategy=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("-"*40)
print('Cross-Validation Results:')
print("-"*40)

svm_cv_scores=cross_val_score(svm_pipeline, x_train, y_train, cv=cv_strategy)
svm_cv_mean= svm_cv_scores.mean()
svm_cv_std= svm_cv_scores.std()
print(f"Support Vector Machine Cross-Validation Scores:{svm_cv_mean:.3f} +/- {svm_cv_std:.3f} (Scores: {svm_cv_scores.round(2)})")

knn_cv_scores=cross_val_score(knn_pipeline, x_train, y_train, cv=cv_strategy)
knn_cv_mean= knn_cv_scores.mean()
knn_cv_std= knn_cv_scores.std()
print(f"KNN Cross-Validation Scores:{knn_cv_mean:.3f} +/- {knn_cv_std:.3f} (Scores: {knn_cv_scores.round(2)})")

rf_cv_scores=cross_val_score(rf, x_train, y_train, cv=cv_strategy)
rf_cv_mean=rf_cv_scores.mean()
rf_cv_std=rf_cv_scores.std()
print(f"Random Forest Cross-Validation Scores:{rf_cv_mean:.3f} +/- {rf_cv_std:.3f} (Scores: {rf_cv_scores.round(2)})")

print("-"*40)
print ("\nFinal Model Evaluation On Test Set:")
print("-"*40)

models={
    'SVM': svm_pipeline,
    'KNN': knn_pipeline,
    'Random Forest': rf
}

results={}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_prediction = model.predict(x_test)

    acc = accuracy_score(y_test, y_prediction)
    results[name] = acc

    print(f"\n{name}:")
    print(f"Test Accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_prediction))

print('\n' + '='*60)
print('FINAL COMPARISON')
print('='*60)

for name, acc in results.items():
    print(f"{name}: {acc:.3f}")

best_model = max(results, key=results.get)
print(f"\nBest Model: {best_model} ({results[best_model]:.3f})")

plt.figure(figsize=(10, 6))
models_list = list(results.keys())
accuracies = list(results.values())

plt.bar(models_list, accuracies, color=['red', 'blue', 'green'])
plt.ylim([0, 1.0])
plt.ylabel('Accuracy')
plt.title('Model Comparison - Customer Churn Prediction')


for i, v in enumerate(accuracies):
    plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()