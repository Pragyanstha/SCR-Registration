from .scrnet import LitSCRNet
from .posenet import LitPoseNet
from .dflnet import LitDFLNet

MODELS = {
    "scrnet": LitSCRNet,
    "posenet": LitPoseNet,
    "dflnet": LitDFLNet
}

def get_model(model_name):
    try:
        model = MODELS[model_name]
    except Exception as e:
        raise Exception("Could not find model")
    return model
