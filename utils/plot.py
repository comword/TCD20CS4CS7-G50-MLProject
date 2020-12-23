import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, confusion_matrix

def plot_roc(models, X_train, X_test, y_train, y_test):
    plt.figure(figsize=(7,5))
    for model in models:
        model.fit(X_train, y_train)
        if callable(getattr(model, "predict_proba", None)):
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        elif callable(getattr(model, "decision_function", None)):
            fpr, tpr, _ = roc_curve(y_test, model.decision_function(X_test))
        else:
            raise NotImplementedError
        plt.plot(fpr, tpr, label=model.name())
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--', label='baseline')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()

def plot_confusion_matrix(model, X_test, y_test):
    tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_test)).ravel()
    return (tn, fp, fn, tp)
