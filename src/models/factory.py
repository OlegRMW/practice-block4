from models.ResNet18 import CIFAR_Classifier

MODELS = {
    "resnet": CIFAR_Classifier,
}


def get_model(model_type):
    if model_type not in MODELS:
        raise ValueError(f"Unknown model: {model_type}")

    return MODELS[model_type]()