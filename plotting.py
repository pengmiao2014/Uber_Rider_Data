import numpy as np
import sklearn.learning_curve as curves
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import confusion_matrix


def standard_confusion_matrix(y_true, y_predict):
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_predict)
    return np.array([[tp, fp], [fn, tn]])


def profit_curve(cost_benefit_matrix, probabilities, y_true):
    thresholds = sorted(probabilities, reverse=True)
    profits = []
    for threshold in thresholds:
        y_predict = probabilities > threshold
        confusion_mat = standard_confusion_matrix(y_true, y_predict)
        profit = np.sum(confusion_mat * cost_benefit_matrix) / float(len(y_true))
        profits.append(profit)
    return thresholds, profits


def run_profit_curve(model, costbenefit, X_train, X_test, y_train, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    thresholds, profits = profit_curve(costbenefit, probabilities, y_test)
    return thresholds, profits


def plot_profit_model(model, costbenefit, X_train, X_test, y_train, y_test):
    percentages = np.linspace(0, 100, len(y_test))
    thresholds, profits = run_profit_curve(model,
                                           costbenefit,
                                           X_train, X_test,
                                           y_train, y_test)
    plt.plot(percentages, profits, label=model.__class__.__name__)
    plt.title("Profit Curve")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='best')
    plt.savefig('profit_curve.png')


def find_best_threshold(model, costbenefit, X_train, X_test, y_train, y_test):
    max_threshold = None
    max_profit = None

    thresholds, profits = run_profit_curve(model, costbenefit,
                                           X_train, X_test,
                                           y_train, y_test)
    max_index = np.argmax(profits)
    if profits[max_index] > max_profit:
        max_threshold = thresholds[max_index]
        max_profit = profits[max_index]
    return max_threshold, max_profit