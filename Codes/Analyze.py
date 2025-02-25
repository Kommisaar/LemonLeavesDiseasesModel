import pandas as pd
from torchvision.models import resnet18
if __name__ == '__main__':
    data = pd.read_excel('../Results/Predictions-LemonLeaves.xlsx')

    cols = data.columns
    data["Prediction Label"] = data["Prediction"].apply(lambda x: cols[x])
    data["Actual Label"] = data["Actual"].apply(lambda x: cols[x])
    data["Correct"] = data["Prediction"] == data["Actual"]
    data["Accuracy"] = len(data[data["Prediction"] == data["Actual"]])/len(data)
    print(data[data["Correct"]==False][["Prediction Label", "Actual Label", "Accuracy"]])

print(resnet18())