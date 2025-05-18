import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split, GridSearchCV, learning_curve, StratifiedKFold,
    cross_val_score
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import joblib # For saving and loading model
import os

# Optional: SHAP for model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP library not found. Skipping SHAP analysis. Install with: pip install shap")

# Optional: mlxtend for decision region plots
try:
    from mlxtend.plotting import plot_decision_regions
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    print("mlxtend library not found. Skipping decision region plot. Install with: pip install mlxtend")


# 1. Load and Prepare Data
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("Iris Dataset:")
print(df.head())

X = df.drop('species', axis=1)
y = df['species']
n_classes = len(np.unique(y))
feature_names = X.columns.tolist()
class_names = iris.target_names.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2. Advanced Hyperparameter Tuning with Cost-Complexity Pruning
base_dt = DecisionTreeClassifier(random_state=42)
path = base_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
# Remove the largest alpha, which prunes the tree to a single node
ccp_alphas = ccp_alphas[:-1]
# Ensure non-negative and add 0.0 if not present for unpruned tree
ccp_alphas = ccp_alphas[ccp_alphas >= 0]
if 0.0 not in ccp_alphas:
    ccp_alphas = np.concatenate(([0.0], ccp_alphas))
ccp_alphas = np.unique(ccp_alphas) # Ensure unique values

# Plot accuracy vs alpha for training and testing sets
# This helps visualize the effect of ccp_alpha before GridSearchCV
if len(ccp_alphas) > 1: # Only plot if there's a range of alphas
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
    ax.legend()
    plt.grid()
    plt.show()

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'ccp_alpha': ccp_alphas if len(ccp_alphas) > 0 else [0.0] # Use refined alphas
}

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=stratified_kfold,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=0) # Set to 1 or 2 for more verbosity
grid_search.fit(X_train, y_train)

best_classifier = grid_search.best_estimator_

print("\nBest Hyperparameters found by GridSearchCV:")
print(grid_search.best_params_)

# 3. Model Evaluation
y_pred = best_classifier.predict(X_test)
y_proba = best_classifier.predict_proba(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_report = metrics.classification_report(y_test, y_pred, target_names=class_names)
mcc = metrics.matthews_corrcoef(y_test, y_pred)
balanced_accuracy = metrics.balanced_accuracy_score(y_test, y_pred)

print(f"\nOptimized Model Accuracy: {accuracy * 100:.2f}%")
print(f"Optimized Model Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
print(f"Optimized Model Matthews Correlation Coefficient: {mcc:.3f}")
print("\nOptimized Model Confusion Matrix:")
print(conf_matrix)
print("\nOptimized Model Classification Report:")
print(class_report)

# 4. Advanced Visualizations and Interpretability

# 4.1. Feature Importances
importances = best_classifier.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_feature_names = [feature_names[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances from Optimized Decision Tree")
plt.bar(range(X.shape[1]), importances[indices], color="skyblue", align="center")
plt.xticks(range(X.shape[1]), sorted_feature_names, rotation=45, ha="right")
plt.xlim([-1, X.shape[1]])
plt.ylabel("Importance (Gini Impurity Reduction)")
plt.xlabel("Feature")
plt.tight_layout()
plt.show()

# 4.2. Decision Tree Plot
plt.figure(figsize=(20,12)) # Increased size for clarity
plot_tree(best_classifier,
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True,
          precision=3, # More precision for values/impurity
          fontsize=8)  # Adjusted font size
plt.title(f"Optimized Decision Tree (ccp_alpha={best_classifier.ccp_alpha:.4f})", fontsize=16)
plt.show()

# 4.3. ROC Curves (Multi-class)
y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes))

fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test_binarized[:, i], y_proba[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_binarized.ravel(), y_proba.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

plt.figure(figsize=(10, 7))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
plt.plot(fpr["micro"], tpr["micro"],
         label=f'Micro-average ROC (AUC = {roc_auc["micro"]:0.3f})',
         color='deeppink', linestyle=':', linewidth=3)
plt.plot(fpr["macro"], tpr["macro"],
         label=f'Macro-average ROC (AUC = {roc_auc["macro"]:0.3f})',
         color='navy', linestyle=':', linewidth=3)
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC class {class_names[i]} (AUC = {roc_auc[i]:0.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Multi-class Receiver Operating Characteristic (ROC) Curves', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.5)
plt.show()

# 4.4. Precision-Recall Curves
precision, recall = dict(), dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = metrics.precision_recall_curve(y_test_binarized[:, i], y_proba[:, i])
    average_precision[i] = metrics.average_precision_score(y_test_binarized[:, i], y_proba[:, i])

# Micro-averaged Precision-Recall
precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_test_binarized.ravel(), y_proba.ravel())
average_precision["micro"] = metrics.average_precision_score(y_test_binarized, y_proba, average="micro")

plt.figure(figsize=(10, 7))
plt.step(recall['micro'], precision['micro'], where='post', color='purple',
         label=f'Micro-average PR (AP = {average_precision["micro"]:0.3f})')
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label=f'PR curve of class {class_names[i]} (AP = {average_precision[i]:0.3f})')

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Multi-class Precision-Recall Curves', fontsize=14)
plt.legend(loc="best", fontsize=10) # Changed to "best" for better placement
plt.grid(alpha=0.5)
plt.show()

# 4.5. Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    best_classifier, X, y, cv=stratified_kfold, n_jobs=-1,
    train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.title("Learning Curve", fontsize=14)
plt.xlabel("Training examples", fontsize=12)
plt.ylabel("Score (Accuracy)", fontsize=12)
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best", fontsize=10)
plt.show()

# 4.6. Calibration Curve
plt.figure(figsize=(10, 7))
disp = CalibrationDisplay.from_estimator(best_classifier, X_test, y_test, n_bins=10, strategy='uniform', name='Optimized DT')
plt.title('Calibration Curve (Reliability Diagram)', fontsize=14)
plt.xlabel('Mean Predicted Probability (Positive Class)', fontsize=12)
plt.ylabel('Fraction of Positives (True Frequency)', fontsize=12)
plt.legend(loc="upper left", fontsize=10)
plt.grid(alpha=0.5)
plt.show()

# 4.7. SHAP Analysis (if available)
if SHAP_AVAILABLE:
    print("\nPerforming SHAP Analysis...")
    try:
        # For tree-based models, TreeExplainer is efficient
        explainer = shap.TreeExplainer(best_classifier)
        shap_values = explainer.shap_values(X_test) # For multi-class, this is a list of arrays

        # Summary Plot (beeswarm for overall feature importance and effect)
        # For multi-class, shap_values is a list of arrays (one for each class)
        # We can plot for each class or a combined way if appropriate
        # For simplicity, let's plot for the first class or a specific class.
        # Or, we can use the magnitude of SHAP values if it's about overall importance.
        if isinstance(shap_values, list): # Multi-class case
            print("SHAP summary plot for class 0 (Setosa):")
            shap.summary_plot(shap_values[0], X_test, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary for Class: {class_names[0]}")
            plt.show()
            print("SHAP summary plot for class 1 (Versicolor):")
            shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary for Class: {class_names[1]}")
            plt.show()
            print("SHAP summary plot for class 2 (Virginica):")
            shap.summary_plot(shap_values[2], X_test, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary for Class: {class_names[2]}")
            plt.show()

            # Global bar plot (average |SHAP value|)
            # Need to average absolute SHAP values across all classes for global importance
            shap_values_abs_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            shap.summary_plot(shap_values_abs_mean, X_test, feature_names=feature_names, plot_type="bar", show=False)
            plt.title("Global SHAP Feature Importance (Mean |SHAP value| across classes)")
            plt.show()

            # Force plot for a single prediction (e.g., first instance, for class 0)
            print(f"\nSHAP force plot for first test instance, prediction for class '{class_names[0]}':")
            shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:], feature_names=feature_names, matplotlib=True, show=False)
            plt.title(f"SHAP Force Plot for Instance 0, Class '{class_names[0]}'") # Title doesn't show directly on force_plot with matplotlib=True
            plt.show() # Need to call plt.show() explicitly after matplotlib=True force_plot

        else: # Binary or regression case (though Iris is multi-class)
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
            plt.title("Global SHAP Feature Importance")
            plt.show()
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.title("SHAP Feature Summary")
            plt.show()

    except Exception as e:
        print(f"Could not generate SHAP plots: {e}")
else:
    print("\nSHAP analysis skipped as 'shap' library is not available.")


# 4.8. Decision Regions Plot (mlxtend)
if MLXTEND_AVAILABLE and X_test.shape[1] >= 2:
    print("\nPlotting Decision Regions (using top 2 features)...")
    try:
        # Use the two most important features for visualization
        idx_feature1 = indices[0]
        idx_feature2 = indices[1]
        X_train_plot = X_train.iloc[:, [idx_feature1, idx_feature2]].values
        y_train_plot = y_train.values # mlxtend expects numpy arrays

        # Train a new classifier on these 2 features only for accurate 2D visualization
        classifier_2d = DecisionTreeClassifier(**grid_search.best_params_)
        classifier_2d.fit(X_train_plot, y_train_plot)

        plt.figure(figsize=(10, 7))
        plot_decision_regions(X_train_plot, y_train_plot, clf=classifier_2d, legend=len(np.unique(y_train_plot)))
        plt.xlabel(feature_names[idx_feature1], fontsize=12)
        plt.ylabel(feature_names[idx_feature2], fontsize=12)
        plt.title(f'Decision Regions (Top 2 Features: {feature_names[idx_feature1]} & {feature_names[idx_feature2]})', fontsize=14)
        handles, plt_labels = plt.gca().get_legend_handles_labels() # Get handles and labels
        new_labels = [class_names[int(l)] for l in plt_labels if l.isdigit()] # Map numeric labels to class names
        if len(new_labels) == len(handles): # Ensure mapping was successful
             plt.legend(handles, new_labels, title="Species", fontsize=10)
        else: # Fallback if label mapping fails
            plt.legend(title="Species", fontsize=10)
        plt.grid(alpha=0.3)
        plt.show()
    except Exception as e:
        print(f"Could not plot decision regions with mlxtend: {e}")
else:
    if not MLXTEND_AVAILABLE:
        print("\nmlxtend decision region plot skipped as 'mlxtend' library is not available.")
    else:
        print("\nmlxtend decision region plot skipped as dataset has less than 2 features.")


# 5. Model Persistence (Saving and Loading)
model_filename = 'iris_decision_tree_model.joblib'
joblib.dump(best_classifier, model_filename)
print(f"\nOptimized model saved to {model_filename}")

# Example of loading the model
loaded_model = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# Test loaded model (optional)
y_pred_loaded = loaded_model.predict(X_test)
accuracy_loaded = metrics.accuracy_score(y_test, y_pred_loaded)
print(f"Accuracy of loaded model on test set: {accuracy_loaded * 100:.2f}%")
assert np.array_equal(y_pred, y_pred_loaded), "Predictions from original and loaded model differ!"

# Clean up saved model file (optional, for script cleanliness)
# os.remove(model_filename)
# print(f"Cleaned up {model_filename}")

print("\n--- Analysis Complete ---")