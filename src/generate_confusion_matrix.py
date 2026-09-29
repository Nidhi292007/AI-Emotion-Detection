import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# 1. LOAD EXPERIMENT 28 PREDICTIONS
# ============================================================

file_path = "experiment28_all_predictions.csv"

df = pd.read_csv(file_path)

print("\nCSV Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. EMOTION LABELS
# ============================================================

emotion_names = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]


# ============================================================
# 3. FIND TRUE AND PREDICTED LABEL COLUMNS
# ============================================================

# Possible column names that may exist in the CSV

true_candidates = [
    "y_true",
    "true_label",
    "actual",
    "actual_label",
    "true",
    "Actual",
    "Actual Emotion"
]

pred_candidates = [
    "y_pred",
    "predicted_label",
    "predicted",
    "prediction",
    "Predicted",
    "Predicted Emotion"
]


true_column = None
pred_column = None


for column in true_candidates:
    if column in df.columns:
        true_column = column
        break


for column in pred_candidates:
    if column in df.columns:
        pred_column = column
        break


# ============================================================
# 4. CHECK WHETHER COLUMNS WERE FOUND
# ============================================================

if true_column is None or pred_column is None:

    print("\nCould not automatically identify the label columns.")

    print("\nAvailable columns are:")
    for column in df.columns:
        print(" -", column)

    raise ValueError(
        "\nPlease check the CSV column names and update "
        "true_column and pred_column in the code."
    )


print("\nActual label column   :", true_column)
print("Predicted label column:", pred_column)


# ============================================================
# 5. GET ACTUAL AND PREDICTED VALUES
# ============================================================

y_true = df[true_column]
y_pred = df[pred_column]


# ============================================================
# 6. HANDLE NUMERIC LABELS
# ============================================================

# If labels are 0,1,2,3,4,5 convert them into emotion names.

if pd.api.types.is_numeric_dtype(y_true):

    label_numbers = [0, 1, 2, 3, 4, 5]

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=label_numbers
    )

    display_labels = emotion_names


# ============================================================
# 7. HANDLE TEXT EMOTION LABELS
# ============================================================

else:

    # Convert labels to consistent title case
    y_true = y_true.astype(str).str.strip().str.title()
    y_pred = y_pred.astype(str).str.strip().str.title()

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=emotion_names
    )

    display_labels = emotion_names


# ============================================================
# 8. PRINT CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\n                 Predicted")
print(
    f"{'Actual':<12}"
    f"{'Sadness':>10}"
    f"{'Joy':>10}"
    f"{'Love':>10}"
    f"{'Anger':>10}"
    f"{'Fear':>10}"
    f"{'Surprise':>10}"
)

for i, emotion in enumerate(emotion_names):

    print(
        f"{emotion:<12}"
        f"{cm[i][0]:>10}"
        f"{cm[i][1]:>10}"
        f"{cm[i][2]:>10}"
        f"{cm[i][3]:>10}"
        f"{cm[i][4]:>10}"
        f"{cm[i][5]:>10}"
    )


# ============================================================
# 9. CALCULATE ACCURACY FROM CONFUSION MATRIX
# ============================================================

correct_predictions = cm.trace()

total_predictions = cm.sum()

accuracy = correct_predictions / total_predictions * 100

print("\n" + "=" * 60)
print("PERFORMANCE")
print("=" * 60)

print("Total test samples :", total_predictions)
print("Correct predictions:", correct_predictions)
print("Incorrect predictions:", total_predictions - correct_predictions)
print(f"Accuracy            : {accuracy:.2f}%")


# ============================================================
# 10. GENERATE CONFUSION MATRIX FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(9, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=True
)


# ============================================================
# 11. FORMAT THE FIGURE
# ============================================================

ax.set_title(
    "Confusion Matrix - Complement Naïve Bayes",
    fontsize=16,
    fontweight="bold",
    pad=15
)

ax.set_xlabel(
    "Predicted Emotion",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "Actual Emotion",
    fontsize=12,
    fontweight="bold"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()


# ============================================================
# 12. SAVE HIGH-RESOLUTION IMAGE
# ============================================================

output_file = "confusion_matrix_experiment28.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nConfusion matrix saved as:")
print(output_file)


# ============================================================
# 13. DISPLAY
# ============================================================

plt.show()