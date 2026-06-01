from models.transfer import ProtocolBModel, ProtocolAModel

from .simplenn import SimpleNN
from .wmodel import WorldModel

def make_model(model_name, **kwargs):
    if model_name == "WorldModel":
        
        return WorldModel(SimpleNN, **kwargs)
    else:
        raise ValueError("Model Unavailable")