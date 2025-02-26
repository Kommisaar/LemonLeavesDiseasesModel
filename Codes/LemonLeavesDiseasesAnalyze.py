import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn

if __name__ == '__main__':
    data = pd.read_excel('../Results/Predictions-LemonLeaves.xlsx')

    cols = data.drop(["Prediction", "Actual"], axis=1).columns
    report = classification_report(data["Actual"], data["Prediction"], target_names=cols)
    print(report)

    matrix = confusion_matrix(data["Actual"], data["Prediction"])
    plt.figure(figsize=(12, 12))
    seaborn.heatmap(matrix, annot=True, fmt="d", xticklabels=cols, yticklabels=cols, cmap="Blues")
    plt.savefig("../Results/ConfusionMatrix-LemonLeaves.png")
    plt.show()
