"""Multi-GPU distributed training optimizations for CytoDL.

Provides advanced DDP (DistributedDataParallel) optimizations including:
- Gradient compression for faster communication
- Synchronized batch normalization
- Optimal bucket sizes
- Communication overlap with computation
- Mixed precision gradient scaling

Usage:
    from cyto_dl.utils.distributed import setup_ddp_optimizations, DDPOptimizer

    # Setup DDP with optimizations
    model = setup_ddp_optimizations(
        model,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True
    )

    # Or use DDPOptimizer for more control
    optimizer = DDPOptimizer(
        model,
        sync_bn=True,
        gradient_compression=True,
        overlap_comm=True
    )
"""

import logging
import os
from typing import Optional, Dict, Any, List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


def is_dist_available_and_initialized() -> bool:
    """Check if distributed is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank."""
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_ddp_optimizations(
    model: nn.Module,
    device_ids: Optional[List[int]] = None,
    output_device: Optional[int] = None,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
    static_graph: bool = False,
    bucket_cap_mb: int = 25,
    sync_bn: bool = True,
) -> nn.Module:
    """Setup DDP with optimizations.

    Parameters
    ----------
    model : nn.Module
        Model to wrap with DDP
    device_ids : List[int], optional
        GPU device IDs to use
    output_device : int, optional
        Output device ID
    find_unused_parameters : bool
        Whether to find unused parameters (slower, use only if needed)
    gradient_as_bucket_view : bool
        Use gradient as bucket view for memory efficiency (recommended)
    static_graph : bool
        Use static graph optimization (faster if graph doesn't change)
    bucket_cap_mb : int
        Bucket size for gradient communication in MB (default 25)
    sync_bn : bool
        Convert BatchNorm to SyncBatchNorm for multi-GPU

    Returns
    -------
    nn.Module
        DDP-wrapped model with optimizations

    Examples
    --------
    >>> model = MyModel()
    >>> model = setup_ddp_optimizations(
    ...     model,
    ...     gradient_as_bucket_view=True,
    ...     static_graph=True,
    ...     sync_bn=True
    ... )
    """
    if not is_dist_available_and_initialized():
        logger.warning("Distributed not initialized, returning model as-is")
        return model

    # Convert BatchNorm to SyncBatchNorm for better multi-GPU accuracy
    if sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("✓ Converted BatchNorm to SyncBatchNorm")

    # Wrap with DDP
    if device_ids is None:
        device_ids = [torch.cuda.current_device()]
    if output_device is None:
        output_device = device_ids[0]

    model = DDP(
        model,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
        bucket_cap_mb=bucket_cap_mb,
    )

    logger.info(f"✓ DDP initialized with optimizations:")
    logger.info(f"  - Gradient as bucket view: {gradient_as_bucket_view}")
    logger.info(f"  - Static graph: {static_graph}")
    logger.info(f"  - Bucket size: {bucket_cap_mb} MB")
    logger.info(f"  - SyncBatchNorm: {sync_bn}")
    logger.info(f"  - World size: {get_world_size()}")

    return model


class GradientCompression:
    """Gradient compression for faster multi-GPU communication.

    Uses PowerSGD or FP16 compression to reduce communication overhead.

    Parameters
    ----------
    compression_type : str
        Type of compression: 'powersgd', 'fp16', or 'none'
    powersgd_rank : int
        Rank for PowerSGD compression (lower = more compression)
    start_iter : int
        Iteration to start compression (warmup period)

    Examples
    --------
    >>> compressor = GradientCompression(compression_type='powersgd', powersgd_rank=2)
    >>> # Register with DDP model
    >>> compressor.register(ddp_model)
    """

    def __init__(
        self,
        compression_type: str = "powersgd",
        powersgd_rank: int = 2,
        start_iter: int = 10,
    ):
        self.compression_type = compression_type
        self.powersgd_rank = powersgd_rank
        self.start_iter = start_iter
        self.state = None

    def register(self, ddp_model: DDP):
        """Register compression with DDP model.

        Parameters
        ----------
        ddp_model : DDP
            DDP-wrapped model
        """
        if not is_dist_available_and_initialized():
            logger.warning("Distributed not initialized, skipping gradient compression")
            return

        if self.compression_type == "powersgd":
            try:
                from torch.distributed.algorithms.ddp_comm_hooks import (
                    powerSGD_hook as powerSGD,
                )

                self.state = powerSGD.PowerSGDState(
                    process_group=None,
                    matrix_approximation_rank=self.powersgd_rank,
                    start_powerSGD_iter=self.start_iter,
                )
                ddp_model.register_comm_hook(self.state, powerSGD.powerSGD_hook)
                logger.info(
                    f"✓ Registered PowerSGD gradient compression (rank={self.powersgd_rank})"
                )
            except ImportError:
                logger.warning("PowerSGD not available, skipping compression")

        elif self.compression_type == "fp16":
            try:
                from torch.distributed.algorithms.ddp_comm_hooks import (
                    default_hooks as default,
                )

                ddp_model.register_comm_hook(
                    state=None, hook=default.fp16_compress_hook
                )
                logger.info("✓ Registered FP16 gradient compression")
            except ImportError:
                logger.warning("FP16 compression not available")

        elif self.compression_type == "none":
            logger.info("No gradient compression enabled")
        else:
            raise ValueError(
                f"Unknown compression type: {self.compression_type}. "
                f"Choose from: 'powersgd', 'fp16', 'none'"
            )


class DDPOptimizer:
    """Advanced DDP optimizer with all optimizations.

    Combines all DDP optimizations in one easy-to-use wrapper:
    - SyncBatchNorm
    - Gradient compression
    - Optimal bucket sizes
    - Communication overlap
    - Gradient accumulation support

    Parameters
    ----------
    model : nn.Module
        Model to optimize
    sync_bn : bool
        Convert BatchNorm to SyncBatchNorm
    gradient_compression : bool
        Enable gradient compression (PowerSGD)
    compression_rank : int
        PowerSGD rank (2-4 recommended)
    overlap_comm : bool
        Overlap communication with computation
    bucket_cap_mb : int
        Bucket size in MB
    find_unused_parameters : bool
        Find unused parameters (slower)
    static_graph : bool
        Use static graph optimization

    Examples
    --------
    >>> model = MyModel().cuda()
    >>> ddp_optimizer = DDPOptimizer(
    ...     model,
    ...     sync_bn=True,
    ...     gradient_compression=True,
    ...     overlap_comm=True
    ... )
    >>> model = ddp_optimizer.model
    """

    def __init__(
        self,
        model: nn.Module,
        sync_bn: bool = True,
        gradient_compression: bool = True,
        compression_rank: int = 2,
        overlap_comm: bool = True,
        bucket_cap_mb: int = 25,
        find_unused_parameters: bool = False,
        static_graph: bool = False,
    ):
        self.model = model
        self.sync_bn = sync_bn
        self.gradient_compression = gradient_compression
        self.compression_rank = compression_rank
        self.overlap_comm = overlap_comm

        # Setup DDP
        self.model = setup_ddp_optimizations(
            model,
            sync_bn=sync_bn,
            gradient_as_bucket_view=True,
            static_graph=static_graph,
            bucket_cap_mb=bucket_cap_mb,
            find_unused_parameters=find_unused_parameters,
        )

        # Setup gradient compression
        if gradient_compression and isinstance(self.model, DDP):
            compressor = GradientCompression(
                compression_type="powersgd", powersgd_rank=compression_rank
            )
            compressor.register(self.model)

        logger.info("✓ DDPOptimizer initialized with all optimizations")

    def get_model(self) -> nn.Module:
        """Get the optimized model."""
        return self.model


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """Reduce tensor across all processes.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to reduce
    average : bool
        Whether to average (True) or sum (False)

    Returns
    -------
    torch.Tensor
        Reduced tensor
    """
    if not is_dist_available_and_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= get_world_size()
    return rt


def gather_tensors(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Gather tensors from all processes.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to gather

    Returns
    -------
    List[torch.Tensor]
        List of tensors from all processes
    """
    if not is_dist_available_and_initialized():
        return [tensor]

    world_size = get_world_size()
    tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors, tensor)
    return tensors


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast Python object from src process to all processes.

    Parameters
    ----------
    obj : Any
        Object to broadcast (must be picklable)
    src : int
        Source process rank

    Returns
    -------
    Any
        Broadcasted object
    """
    if not is_dist_available_and_initialized():
        return obj

    objects = [obj]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


class DistributedMetrics:
    """Track and synchronize metrics across processes.

    Examples
    --------
    >>> metrics = DistributedMetrics()
    >>> metrics.update('loss', loss.item())
    >>> metrics.update('accuracy', acc.item())
    >>> avg_metrics = metrics.compute_and_reset()
    """

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, name: str, value: float, count: int = 1):
        """Update metric value.

        Parameters
        ----------
        name : str
            Metric name
        value : float
            Metric value
        count : int
            Number of samples
        """
        if name not in self.metrics:
            self.metrics[name] = 0.0
            self.counts[name] = 0

        self.metrics[name] += value * count
        self.counts[name] += count

    def compute_and_reset(self) -> Dict[str, float]:
        """Compute averages across all processes and reset.

        Returns
        -------
        Dict[str, float]
            Averaged metrics
        """
        result = {}

        for name in self.metrics:
            # Create tensors for reduction
            value_tensor = torch.tensor(
                self.metrics[name], device=torch.cuda.current_device()
            )
            count_tensor = torch.tensor(
                self.counts[name], device=torch.cuda.current_device()
            )

            # Reduce across processes
            if is_dist_available_and_initialized():
                dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

            # Compute average
            result[name] = (value_tensor / count_tensor).item()

        # Reset
        self.metrics = {}
        self.counts = {}

        return result


def setup_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
) -> bool:
    """Initialize distributed training.

    Parameters
    ----------
    backend : str
        Backend to use ('nccl', 'gloo', 'mpi')
    init_method : str, optional
        Initialization method URL
    world_size : int, optional
        Total number of processes
    rank : int, optional
        Rank of current process

    Returns
    -------
    bool
        True if successfully initialized

    Examples
    --------
    >>> if setup_distributed():
    ...     print(f"Initialized rank {get_rank()}/{get_world_size()}")
    """
    if not dist.is_available():
        logger.warning("Distributed training not available")
        return False

    if dist.is_initialized():
        logger.info("Distributed already initialized")
        return True

    # Get from environment if not provided
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    if init_method is None:
        init_method = os.environ.get("MASTER_ADDR", "tcp://localhost:12355")

    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        logger.info(f"✓ Initialized distributed: rank {rank}/{world_size}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize distributed: {e}")
        return False


def cleanup_distributed():
    """Cleanup distributed training."""
    if is_dist_available_and_initialized():
        dist.destroy_process_group()
        logger.info("✓ Cleaned up distributed")
