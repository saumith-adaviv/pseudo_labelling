def get_model_registry():
    from models.bioclip import BioCLIPModel
    from models.dinov2 import DINOv2Model
    from models.siglip import SigLIPModel
    from models.inat_eva02 import iNatEVA02Model
    from models.plantclef import PlantCLEFModel

    return {
        "bioclip2":   BioCLIPModel,
        "dinov2":     DINOv2Model,
        "siglip2":    SigLIPModel,
        "inat_eva02": iNatEVA02Model,
        "plantclef":  PlantCLEFModel,
    }

MODEL_REGISTRY = get_model_registry()
