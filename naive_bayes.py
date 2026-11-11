import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Numeric features: modelled with a normal (Gaussian) distribution
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

categorical_features = ["sex", "cp", "fbs", "restecg", "exang",
                        "slope", "ca", "thal"]

all_features = numeric_features + categorical_features

print("Train size:", len(train), "| Test size:", len(test))


# ---------------------------------------------------------------
# 2. Probability of a numeric value (normal distribution)
# ---------------------------------------------------------------

def gaussian_probability(x, mean, std):
    """Height of the bell curve at x. Closer to the class mean = higher."""
    left = 1 / (std * np.sqrt(2 * np.pi))
    right = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return left * right


# ---------------------------------------------------------------
# 3. Training: collect statistics for each class
# ---------------------------------------------------------------

def train_model(data):
    model = {"prior": {}, "mean": {}, "std": {}, "cat": {}, "unseen": {}}

    for label in [0, 1]:
        part = data[data["target"] == label]        # patients of this class

        # prior: how common is this class?
        model["prior"][label] = len(part) / len(data)

        # numeric features: mean and standard deviation inside the class
        model["mean"][label] = {}
        model["std"][label] = {}
        for feature in numeric_features:
            model["mean"][label][feature] = part[feature].mean()
            model["std"][label][feature] = part[feature].std() + 1e-6

        # categorical features: share of each value inside the class.
        # Laplace smoothing (+1) avoids probability 0 for rare values.
        model["cat"][label] = {}
        model["unseen"][label] = {}
        for feature in categorical_features:
            n_values = data[feature].nunique()
            counts = part[feature].value_counts()

            probs = {}
            for value in data[feature].unique():
                probs[value] = (counts.get(value, 0) + 1) / (len(part) + n_values)

            model["cat"][label][feature] = probs
            model["unseen"][label][feature] = 1 / (len(part) + n_values)

    return model


# ---------------------------------------------------------------
# 4. Prediction
# ---------------------------------------------------------------

def class_score(row, label, model):
    """prior * P(f1|class) * P(f2|class) * ... for one patient and one class"""
    score = model["prior"][label]

    for feature in numeric_features:
        score = score * gaussian_probability(row[feature],
                                             model["mean"][label][feature],
                                             model["std"][label][feature])

    for feature in categorical_features:
        probs = model["cat"][label][feature]
        score = score * probs.get(row[feature], model["unseen"][label][feature])

    return score


def predict_one(row, model):
    score_healthy = class_score(row, 0, model)
    score_sick = class_score(row, 1, model)

    probability_sick = score_sick / (score_healthy + score_sick)

    label = 1 if score_sick > score_healthy else 0
    return label, probability_sick


def predict(data, model):
    labels = []
    probabilities = []
    for _, row in data.iterrows():
        label, prob = predict_one(row, model)
        labels.append(label)
        probabilities.append(prob)
    return np.array(labels), np.array(probabilities)


# ---------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------

def confusion_counts(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def calculate_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return accuracy, precision, recall, f1


def auc_score(y_true, probabilities):
    """Chance that a random sick patient gets a higher probability
    than a random healthy patient (ties count as half)."""
    sick = probabilities[y_true == 1]
    healthy = probabilities[y_true == 0]
    wins = 0
    for s in sick:
        for h in healthy:
            if s > h:
                wins += 1
            elif s == h:
                wins += 0.5
    return wins / (len(sick) * len(healthy))


# ---------------------------------------------------------------
# 6. Train and test
# ---------------------------------------------------------------
model = train_model(train)

y_test = test["target"].values
y_pred, prob_sick = predict(test, model)

accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
auc = auc_score(y_test, prob_sick)
tp, tn, fp, fn = confusion_counts(y_test, y_pred)

print("\n===== RESULTS ON THE TEST SET =====")
print("Accuracy :", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall   :", round(recall, 3))
print("F1 score :", round(f1, 3))
print("AUC      :", round(auc, 3))
print("TP =", tp, " TN =", tn, " FP =", fp, " FN =", fn)


# ---------------------------------------------------------------
# 7. Confusion matrix and ROC curve
# ---------------------------------------------------------------

matrix = np.array([[tn, fp],
                   [fn, tp]])

plt.figure(figsize=(4, 4))
plt.imshow(matrix, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, matrix[i, j], ha="center", va="center", fontsize=16)
plt.xticks([0, 1], ["Pred. healthy", "Pred. sick"])
plt.yticks([0, 1], ["Actually healthy", "Actually sick"])
plt.title("Confusion matrix")
plt.show()

fpr_list = []
tpr_list = []
for t in np.linspace(0, 1, 101):
    pred_t = (prob_sick >= t).astype(int)
    tp_t, tn_t, fp_t, fn_t = confusion_counts(y_test, pred_t)
    tpr_list.append(tp_t / (tp_t + fn_t))
    fpr_list.append(fp_t / (fp_t + tn_t))

plt.figure(figsize=(5, 5))
plt.plot(fpr_list, tpr_list, label="Naive Bayes (AUC = " + str(round(auc, 2)) + ")")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random guessing")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate (recall)")
plt.title("ROC curve")
plt.legend()
plt.show()


# ---------------------------------------------------------------
# 8. Cross-validation (5 folds)
# ---------------------------------------------------------------

full = pd.concat([train, test]).reset_index(drop=True)
full = full.sample(frac=1, random_state=1).reset_index(drop=True)

folds = np.array_split(np.arange(len(full)), 5)
fold_accuracies = []

print("\n===== 5-FOLD CROSS-VALIDATION =====")
for i in range(5):
    test_part = full.iloc[folds[i]]
    train_part = full.drop(full.index[folds[i]])

    fold_model = train_model(train_part)
    fold_pred, _ = predict(test_part, fold_model)
    fold_acc = np.mean(fold_pred == test_part["target"].values)

    fold_accuracies.append(fold_acc)
    print("Fold", i + 1, "accuracy:", round(fold_acc, 3))

print("Mean accuracy:", round(np.mean(fold_accuracies), 3),
      "+/-", round(np.std(fold_accuracies), 3))


# ---------------------------------------------------------------
# 9. Feature importance (permutation method)
# ---------------------------------------------------------------

np.random.seed(0)
baseline_accuracy = np.mean(y_pred == y_test)
importance = {}

for feature in all_features:
    drops = []
    for _ in range(30):
        shuffled = test.copy()
        shuffled[feature] = np.random.permutation(shuffled[feature].values)
        shuffled_pred, _ = predict(shuffled, model)
        drops.append(baseline_accuracy - np.mean(shuffled_pred == y_test))
    importance[feature] = np.mean(drops)

importance = pd.Series(importance).sort_values()

print("\n===== FEATURE IMPORTANCE (accuracy drop) =====")
print(importance.sort_values(ascending=False).round(3))

plt.figure(figsize=(7, 5))
plt.barh(importance.index, importance.values, color="tab:blue")
plt.xlabel("Drop in accuracy when the feature is shuffled")
plt.title("Feature importance")
plt.show()


# ---------------------------------------------------------------
# 10. One patient explained step by step
# ---------------------------------------------------------------

patient = test.iloc[0]

print("\n===== EXAMPLE: ONE PATIENT =====")
print(patient[all_features].to_string())
print("Real class:", int(patient["target"]))

for label in [0, 1]:
    print("\n--- Class", label, "(healthy)" if label == 0 else "(sick)", "---")
    print("prior =", round(model["prior"][label], 3))
    score = model["prior"][label]

    for feature in numeric_features:
        p = gaussian_probability(patient[feature],
                                 model["mean"][label][feature],
                                 model["std"][label][feature])
        score = score * p
        print(feature, "-> P =", round(p, 4))

    for feature in categorical_features:
        probs = model["cat"][label][feature]
        p = probs.get(patient[feature], model["unseen"][label][feature])
        score = score * p
        print(feature, "-> P =", round(p, 4))

    print("Final score for class", label, "=", score)

label, prob = predict_one(patient, model)
print("\nPrediction:", label, "| Probability of being sick:", round(prob, 3))
