__version__ = "0.6.1"


# silence bio packages warnings
import logging
import warnings

logging.getLogger("ome_zarr").setLevel(logging.WARNING)
logging.getLogger("ome_zarr.reader").setLevel(logging.WARNING)
logging.getLogger("bfio.init").setLevel(logging.ERROR)
logging.getLogger("bfio.backends").setLevel(logging.ERROR)
logging.getLogger("xmlschema").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Allow MONAI enums in torch.load(weights_only=True) (PyTorch >= 2.6 default)
import torch

if hasattr(torch.serialization, "add_safe_globals"):
    from monai.data.meta_tensor import MetaTensor
    from monai.utils.enums import TraceKeys

    torch.serialization.add_safe_globals([MetaTensor, TraceKeys])
