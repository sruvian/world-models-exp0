from .simplenn import SimpleNN
from .wmodel import WorldModel

def make_model(model_name, **kwargs):
    if model_name != "WorldModel":
        raise ValueError("Model Unavailable")
    return WorldModel(SimpleNN, **kwargs)