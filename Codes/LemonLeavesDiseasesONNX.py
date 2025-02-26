import torch
import onnxruntime as ort

from LemonLeavesDiseasesModel import LemonLeavesDiseasesModel
from torchvision.transforms import v2
from PIL import Image
from torch.nn import Softmax


def export_onnx_model():
    model = LemonLeavesDiseasesModel()
    model.load_state_dict(torch.load("../ModelWeights/LastModel-LemonLeaves.pth", map_location=torch.device('cpu')))
    model.eval()

    torch.onnx.export(
        model,
        torch.randn(1, 3, 480, 480),
        "../ModelWeights/LastModel-LemonLeaves.onnx",
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )


class ONNXModel():
    def __init__(self, model_path):
        self.classes_types = ["Anthracnose", "Bacterial Blight", "Citrus Canker", "Curl Virus", "Deficiency Leaf",
                              "Dry Leaf", "Healthy Leaf", "Sooty Mould", "Spider Mites"]
        self.sess = ort.InferenceSession("../ModelWeights/LastModel-LemonLeaves.onnx")
        self.transformer = v2.Compose([
            v2.Resize(size=(480, 480)),
            v2.Grayscale(num_output_channels=3),
            v2.PILToTensor(),
            v2.ToDtype(dtype=torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        image = self.transformer(Image.open(image_path))

        probs = Softmax(dim=1)(torch.from_numpy(self.sess.run(None, {"input": image.unsqueeze(0).numpy()})[0])).numpy()
        return self.classes_types[probs.argmax()], probs.max()


if __name__ == '__main__':
    model = ONNXModel("../ModelWeights/LastModel-LemonLeaves.onnx")
    print(model.predict("../Data/New Images/炭疽病1.jpg"))
