import os

os.makedirs("figures", exist_ok=True)
matplotlib_cache = os.path.join(os.getcwd(), ".matplotlib")
os.makedirs(matplotlib_cache, exist_ok=True)
os.environ["MPLCONFIGDIR"] = matplotlib_cache

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_and_show(name):
    plt.savefig("figures/" + name + ".png", dpi=150, bbox_inches="tight")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close()


# ---------------------------------------------------------------
# 1. Load the data and settings
# ---------------------------------------------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
full = pd.concat([train, test]).reset_index(drop=True)

numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang",
                        "slope", "ca", "thal"]
all_features = numeric_features + categorical_features

# Features with a long right tail, used in variant B
log_features = ["trestbps", "chol", "oldpeak"]

REPEATS = 10
MIN_IMPORTANCE = 0.01

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
def train_model(data, numeric, categorical):
    model = {"numeric": numeric, "categorical": categorical,
             "prior": {}, "mean": {}, "std": {}, "cat": {}, "unseen": {}}

    for label in [0, 1]:
        part = data[data["target"] == label]

        model["prior"][label] = len(part) / len(data)

        model["mean"][label] = {}
        model["std"][label] = {}
        for feature in numeric:
            model["mean"][label][feature] = part[feature].mean()
            model["std"][label][feature] = part[feature].std() + 1e-6

        model["cat"][label] = {}
        model["unseen"][label] = {}
        for feature in categorical:
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

    for feature in model["numeric"]:
        score = score * gaussian_probability(row[feature],
                                             model["mean"][label][feature],
                                             model["std"][label][feature])

    for feature in model["categorical"]:
        probs = model["cat"][label][feature]
        score = score * probs.get(row[feature], model["unseen"][label][feature])

    return score


def probability_sick(row, model):
    score_healthy = class_score(row, 0, model)
    score_sick = class_score(row, 1, model)
    total = score_healthy + score_sick
    if total == 0:
        return 0.5
    return score_sick / total


def predict(data, model, threshold=0.5):
    """Returns the predicted labels and the probability of being sick.
    A patient is called sick if the probability is above the threshold."""
    probabilities = []
    for row in data.to_dict("records"):
        probabilities.append(probability_sick(row, model))
    probabilities = np.array(probabilities)
    labels = (probabilities > threshold).astype(int)
    return labels, probabilities


# ---------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------
def confusion_counts(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def f_beta(precision, recall, beta):
    """beta = 1 is the normal F1. beta = 2 values recall more than precision."""
    if precision + recall == 0:
        return 0
    return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)


def calculate_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f_beta(precision, recall, 1)
    return accuracy, precision, recall, f1


def auc_score(y_true, probabilities):
    """Chance that a random sick patient gets a higher probability
    than a random healthy patient (ties count as half)."""
    sick = probabilities[y_true == 1]
    healthy = probabilities[y_true == 0]
    wins = (sick[:, None] > healthy[None, :]).sum()
    ties = (sick[:, None] == healthy[None, :]).sum()
    return (wins + 0.5 * ties) / (len(sick) * len(healthy))


# ---------------------------------------------------------------
# 6. Preparing the data for a model variant
# ---------------------------------------------------------------

def prepare(train_part, test_part, config):
    """Apply the config to the train and test part.
    Bin limits are calculated from the train part only."""
    numeric = [f for f in numeric_features if f not in config["drop"]]
    categorical = [f for f in categorical_features if f not in config["drop"]]
    train_p = train_part.copy()
    test_p = test_part.copy()

    if config["transform"] == "log":
        for feature in log_features:
            if feature in numeric:
                train_p[feature] = np.log1p(train_p[feature])
                test_p[feature] = np.log1p(test_p[feature])

    if config["transform"] == "bins":
        for feature in numeric:
            limits = np.quantile(train_p[feature], [0.25, 0.5, 0.75])
            train_p[feature] = np.digitize(train_p[feature], limits)
            test_p[feature] = np.digitize(test_p[feature], limits)
        categorical = categorical + numeric
        numeric = []

    return train_p, test_p, numeric, categorical


# ---------------------------------------------------------------
# 7. Cross-validation tools
# ---------------------------------------------------------------
def make_folds(y, k, seed):
    """Split the patient numbers into k folds.
    Healthy and sick patients are spread evenly over the folds."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for label in [0, 1]:
        indexes = np.where(y == label)[0]
        rng.shuffle(indexes)
        for i, part in enumerate(np.array_split(indexes, k)):
            folds[i].extend(part)
    return [np.array(f) for f in folds]


def cross_validate(data, config, repeats=REPEATS, k=5):
    """Run k-fold cross-validation several times with different shuffles."""
    y = data["target"].values
    fold_accuracies = []
    repeat_results = []
    probability_sum = np.zeros(len(data))

    for r in range(repeats):
        folds = make_folds(y, k, seed=r)
        out_of_fold = np.zeros(len(data))

        for i in range(k):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

            train_p, test_p, numeric, categorical = prepare(
                data.iloc[train_idx], data.iloc[test_idx], config)
            model = train_model(train_p, numeric, categorical)
            labels, probs = predict(test_p, model)

            out_of_fold[test_idx] = probs
            fold_accuracies.append(np.mean(labels == y[test_idx]))

        labels = (out_of_fold > 0.5).astype(int)
        accuracy, precision, recall, f1 = calculate_metrics(y, labels)
        repeat_results.append([accuracy, precision, recall, f1,
                               auc_score(y, out_of_fold)])
        probability_sum += out_of_fold

    mean = np.mean(repeat_results, axis=0)
    return {"name": config["name"],
            "accuracy": mean[0], "fold_std": np.std(fold_accuracies),
            "precision": mean[1], "recall": mean[2],
            "f1": mean[3], "auc": mean[4],
            "probabilities": probability_sum / repeats}


def cv_importance(data, config, repeats=3, shuffles=5, k=5):
    """Permutation importance with cross-validation.
    For each fold we shuffle one feature in the fold's test part and see
    how much the accuracy drops. A big drop = the model needs that feature."""
    y = data["target"].values
    rng = np.random.default_rng(0)
    drops = {}

    for r in range(repeats):
        folds = make_folds(y, k, seed=100 + r)
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

            train_p, test_p, numeric, categorical = prepare(
                data.iloc[train_idx], data.iloc[test_idx], config)
            model = train_model(train_p, numeric, categorical)

            y_fold = y[test_idx]
            labels, _ = predict(test_p, model)
            baseline = np.mean(labels == y_fold)

            for feature in numeric + categorical:
                fold_drops = []
                for _ in range(shuffles):
                    shuffled = test_p.copy()
                    shuffled[feature] = rng.permutation(shuffled[feature].values)
                    shuffled_labels, _ = predict(shuffled, model)
                    fold_drops.append(baseline - np.mean(shuffled_labels == y_fold))
                drops.setdefault(feature, []).append(np.mean(fold_drops))

    mean_drop = pd.Series({f: np.mean(d) for f, d in drops.items()})
    std_drop = pd.Series({f: np.std(d) for f, d in drops.items()})
    return mean_drop, std_drop


# ---------------------------------------------------------------
# 8. STEP 1: which features have no importance? (train data only)
# ---------------------------------------------------------------
config_all = {"name": "A: all 13 features", "transform": "none", "drop": []}

print("\nCalculating importance of the features on the train data...")
select_mean, select_std = cv_importance(train, config_all)

print("\n===== STEP 1: FEATURE SELECTION (train data only) =====")
print("A feature is kept if shuffling it lowers the accuracy by more than",
      MIN_IMPORTANCE)

dropped = [f for f in all_features if select_mean[f] <= MIN_IMPORTANCE]

# safety: never keep fewer than 4 features
if len(all_features) - len(dropped) < 4:
    best_four = list(select_mean.sort_values(ascending=False).index[:4])
    dropped = [f for f in all_features if f not in best_four]

for feature in select_mean.sort_values(ascending=False).index:
    status = "DROP" if feature in dropped else "keep"
    print(feature.ljust(9), str(round(select_mean[feature], 3)).rjust(7),
          "+/-", round(select_std[feature], 3), "  ", status)

print("\nDropped features:", dropped if dropped else "none")
print("Kept features   :", [f for f in all_features if f not in dropped])


# ---------------------------------------------------------------
# 9. STEP 2: compare the model variants (train data only)
# ---------------------------------------------------------------
configs = [
    config_all,
    {"name": "B: log of skewed features", "transform": "log", "drop": []},
    {"name": "C: numeric features in bins", "transform": "bins", "drop": []},
]
if len(dropped) > 0:
    configs.append({"name": "D: selected features only",
                    "transform": "none", "drop": dropped})

print("\n===== STEP 2: COMPARING MODEL VARIANTS (" + str(REPEATS) +
      " x 5-fold cross-validation on the train data) =====")

results = []
for config in configs:
    results.append(cross_validate(train, config))

table = pd.DataFrame(results).set_index("name")
print(table[["accuracy", "fold_std", "precision", "recall", "f1", "auc"]]
      .round(4).to_string())

majority = max(train["target"].mean(), 1 - train["target"].mean())
print("\nBaseline (predict the biggest class for everybody):", round(majority, 3))

best_index = int(np.argmax([r["accuracy"] for r in results]))
best_config = configs[best_index]
best_result = results[best_index]
print("Chosen variant:", best_config["name"])


# ---------------------------------------------------------------
# 10. STEP 3: tune the decision threshold (train data only)
# ---------------------------------------------------------------

y_train = train["target"].values
train_probs = best_result["probabilities"]

print("\n===== STEP 3: THRESHOLD TUNING =====")
threshold_rows = []
for threshold in np.arange(0.05, 0.81, 0.05):
    labels = (train_probs > threshold).astype(int)
    accuracy, precision, recall, f1 = calculate_metrics(y_train, labels)
    f2 = f_beta(precision, recall, 2)
    threshold_rows.append([round(threshold, 2), accuracy, precision, recall, f2])

threshold_table = pd.DataFrame(threshold_rows,
                               columns=["threshold", "accuracy", "precision",
                                        "recall", "F2"])
print(threshold_table.round(3).to_string(index=False))

best_threshold = threshold_table.loc[threshold_table["F2"].idxmax(), "threshold"]
print("Chosen threshold:", best_threshold)

plt.figure(figsize=(6, 4))
threshold_x = threshold_table["threshold"].to_numpy()
plt.plot(threshold_x, threshold_table["accuracy"].to_numpy(), label="accuracy")
plt.plot(threshold_x, threshold_table["precision"].to_numpy(), label="precision")
plt.plot(threshold_x, threshold_table["recall"].to_numpy(), label="recall")
plt.axvline(best_threshold, color="gray", linestyle="--")
plt.xlabel("Threshold for predicting 'sick'")
plt.title("Effect of the threshold")
plt.legend()
save_and_show("threshold_tuning")


# ---------------------------------------------------------------
# 11. STEP 4: final test (the test set is used only here)
# ---------------------------------------------------------------
train_p, test_p, numeric, categorical = prepare(train, test, best_config)
final_model = train_model(train_p, numeric, categorical)

y_test = test["target"].values
_, prob_sick = predict(test_p, final_model)

print("\n===== STEP 4: RESULTS ON THE TEST SET (" + best_config["name"] + ") =====")
print("AUC:", round(auc_score(y_test, prob_sick), 3))

for threshold in [0.5, best_threshold]:
    labels = (prob_sick > threshold).astype(int)
    accuracy, precision, recall, f1 = calculate_metrics(y_test, labels)
    tp, tn, fp, fn = confusion_counts(y_test, labels)
    print("\nThreshold", threshold)
    print("  Accuracy :", round(accuracy, 3))
    print("  Precision:", round(precision, 3))
    print("  Recall   :", round(recall, 3))
    print("  F1 score :", round(f1, 3))
    print("  TP =", tp, " TN =", tn, " FP =", fp, " FN =", fn)

all_train, all_test, all_numeric, all_categorical = prepare(train, test, config_all)
all_model = train_model(all_train, all_numeric, all_categorical)
all_labels, all_probs = predict(all_test, all_model)
accuracy, precision, recall, f1 = calculate_metrics(y_test, all_labels)
print("\nFor comparison, all 13 features (threshold 0.5):")
print("  Accuracy :", round(accuracy, 3), " Recall:", round(recall, 3),
      " AUC:", round(auc_score(y_test, all_probs), 3))

final_labels = (prob_sick > best_threshold).astype(int)
tp, tn, fp, fn = confusion_counts(y_test, final_labels)
matrix = np.array([[tn, fp],
                   [fn, tp]])

plt.figure(figsize=(4, 4))
plt.imshow(matrix, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, matrix[i, j], ha="center", va="center", fontsize=16)
plt.xticks([0, 1], ["Pred. healthy", "Pred. sick"])
plt.yticks([0, 1], ["Actually healthy", "Actually sick"])
plt.title("Confusion matrix (threshold " + str(best_threshold) + ")")
save_and_show("confusion_matrix")

fpr_list = []
tpr_list = []
for t in np.linspace(0, 1, 101):
    pred_t = (prob_sick >= t).astype(int)
    tp_t, tn_t, fp_t, fn_t = confusion_counts(y_test, pred_t)
    tpr_list.append(tp_t / (tp_t + fn_t))
    fpr_list.append(fp_t / (fp_t + tn_t))

plt.figure(figsize=(5, 5))
plt.plot(fpr_list, tpr_list,
         label="Naive Bayes (AUC = " + str(round(auc_score(y_test, prob_sick), 2)) + ")")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random guessing")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate (recall)")
plt.title("ROC curve")
plt.legend()
save_and_show("roc_curve")


# ---------------------------------------------------------------
# 12. STEP 5: stability check on all patients
# ---------------------------------------------------------------
full_result = cross_validate(full, best_config)

print("\n===== " + str(REPEATS) + " x 5-FOLD CROSS-VALIDATION ON ALL " +
      str(len(full)) + " PATIENTS (" + best_config["name"] + ") =====")
print("Accuracy :", round(full_result["accuracy"], 3),
      "(fold std:", round(full_result["fold_std"], 3), ")")
print("Precision:", round(full_result["precision"], 3))
print("Recall   :", round(full_result["recall"], 3))
print("F1 score :", round(full_result["f1"], 3))
print("AUC      :", round(full_result["auc"], 3))


# ---------------------------------------------------------------
# 13. Feature importance of all 13 features (all patients)
# ---------------------------------------------------------------

print("\nCalculating feature importance on all patients...")
mean_drop, std_drop = cv_importance(full, config_all)

order = mean_drop.sort_values()
print("\n===== FEATURE IMPORTANCE, ALL PATIENTS (accuracy drop, mean +/- std) =====")
for feature in order.index[::-1]:
    print(feature.ljust(9), str(round(mean_drop[feature], 3)).rjust(7),
          "+/-", round(std_drop[feature], 3))

colors = ["lightgray" if f in dropped else "tab:blue" for f in order.index]
plt.figure(figsize=(7, 5))
plt.barh(order.index, order.values, xerr=std_drop[order.index].values,
         color=colors, capsize=3)
plt.xlabel("Drop in accuracy when the feature is shuffled")
plt.title("Feature importance (grey = dropped in step 1)")
save_and_show("feature_importance")


# ---------------------------------------------------------------
# 14. One patient explained step by step
# ---------------------------------------------------------------
patient = test_p.iloc[0]

print("\n===== EXAMPLE: ONE PATIENT (" + best_config["name"] + ") =====")
print(patient[numeric + categorical].to_string())
print("Real class:", int(patient["target"]))

for label in [0, 1]:
    print("\n--- Class", label, "(healthy)" if label == 0 else "(sick)", "---")
    print("prior =", round(final_model["prior"][label], 3))
    score = final_model["prior"][label]

    for feature in numeric:
        p = gaussian_probability(patient[feature],
                                 final_model["mean"][label][feature],
                                 final_model["std"][label][feature])
        score = score * p
        print(feature, "-> P =", round(p, 4))

    for feature in categorical:
        probs = final_model["cat"][label][feature]
        p = probs.get(patient[feature], final_model["unseen"][label][feature])
        score = score * p
        print(feature, "-> P =", round(p, 4))

    print("Final score for class", label, "=", score)

prob = probability_sick(patient, final_model)
print("\nProbability of being sick:", round(prob, 3))
print("Prediction:", 1 if prob > best_threshold else 0)
