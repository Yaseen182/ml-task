from dataOperation.data_loader import load_data
from dataOperation.preprocessing import handle_missing_values, remove_outliers, normalize_data

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




def train_model(X_train, y_train):

    # 1) Base model
    model = RandomForestClassifier(random_state=42)

    # 2) Hyperparameters search space
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2]
    }

    # 3) Grid Search
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,               # 3-fold cross validation
        scoring="accuracy",
        n_jobs=-1,          # faster using all CPU cores
        verbose=1
    )

    # 4) Train
    grid.fit(X_train, y_train)

    # Print best results
    print("\nBest Hyperparameters found:")
    print(grid.best_params_)

    print("\nBest Cross-Validation Accuracy:")
    print(grid.best_score_)

    # Return the best model
    return grid.best_estimator_




def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Test Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()




def main():
    # Load data
    X, y = load_data()

    # Preprocessing
    X = handle_missing_values(X)
    X, y = remove_outliers(X, y)
    X_scaled = normalize_data(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train with GridSearchCV
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_test, y_test)



if __name__ == "__main__":
    main()
