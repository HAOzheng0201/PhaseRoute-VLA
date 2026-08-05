from .config import *
from .model import *


def check_install(cuda: bool = False):
    import torch

    from .version import VERSION

    if cuda:
        assert torch.cuda.is_available(), "CUDA is not available!"
        print("CUDA available")

    print(f"PhaseRoute-VLA v{VERSION} installed (Python package: a1)")
