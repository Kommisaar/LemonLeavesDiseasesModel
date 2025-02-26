import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2

from LemonLeavesDiseasesModel import LemonLeavesDiseasesModel

if __name__ == '__main__':
    image = Image.open("../Data/New Images/炭疽病1.jpg")
    transform = v2.Compose([
        v2.Resize(size=(480, 480)),
        v2.PILToTensor(),
        v2.ToDtype(dtype=torch.float32)
    ])

    model = LemonLeavesDiseasesModel()
    model.load_state_dict(torch.load("../ModelWeights/LastModel-LemonLeaves.pth", map_location=torch.device('cpu')))

    probs = model.predict_proba(transform(image).unsqueeze(0))
    prediction_dataframe = pd.DataFrame(probs, columns=model.classes_types)
    prediction_dataframe["Prediction"] = torch.argmax(probs, dim=1)
    prediction_dataframe["Prediction Label"] = prediction_dataframe["Prediction"].apply(lambda x: model.classes_types[x])

    print(prediction_dataframe)
