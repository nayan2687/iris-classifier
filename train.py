from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")

    cm = confusion_matrix(y_test, y_pred)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Iris Confusion Matrix")
    plt.tight_layout()
    plt.savefig(outputs_dir / "confusion_matrix.png")
    plt.close()

    joblib.dump(model, outputs_dir / "iris_model.joblib")
    print("Model and confusion matrix saved in outputs/")

if __name__ == "__main__":
    main()
