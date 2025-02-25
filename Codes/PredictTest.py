from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from LemonLeavesDiseaseModel import LemonLeavesDiseasesModel
from LemonLeavesDiseasesDataset import lemon_leaves_load_data, LemonLeavesDiseasesDataset

if __name__ == "__main__":

    root_dir = Path("../Data/Original Dataset")
    data, target = lemon_leaves_load_data(root_dir)

    train_data, test_data, train_labels, test_labels = train_test_split(data, target, test_size=0.2, random_state=42)

    class_types = [str(class_name.stem) for class_name in root_dir.iterdir()]
    model = LemonLeavesDiseasesModel(len(class_types))
    model.load_state_dict(torch.load("../ModelWeights/LastModel-LemonLeaves.pth",map_location="cpu"))

    data_loader = DataLoader(LemonLeavesDiseasesDataset(test_data, test_labels), shuffle=True, batch_size=32)

    with torch.no_grad():
        model.eval()
        predictions = []
        probs = []
        actual = []
        for images, labels in data_loader:
            prob = model.predict_proba(images)
            probs.append(prob)
            predictions.append(torch.argmax(prob, dim=1))
            actual.append(labels)
        probs = torch.cat(probs)
        predictions = torch.cat(predictions)
        actual = torch.cat(actual)

        res = pd.DataFrame(probs.numpy(), columns=class_types)

        res['Prediction'] = predictions.numpy()
        res['Actual'] = actual.numpy()

        res.to_excel("../Results/Predictions-LemonLeaves.xlsx", index=False)
