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

# PyTorch 2.6+ defaults torch.load(weights_only=True). Lightning passes that
# through when resuming from a checkpoint, and cyto-dl checkpoints contain
# MONAI MetaTensor metadata (TraceKeys/MetaKeys enums) that must be
# explicitly allowlisted for the safe unpickler.
import torch.serialization
from monai.utils.enums import MetaKeys, TraceKeys

torch.serialization.add_safe_globals([TraceKeys, MetaKeys])
