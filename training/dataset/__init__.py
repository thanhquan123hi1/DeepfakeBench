import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)


from .abstract_dataset import DeepfakeAbstractBaseDataset

try:
    from .I2G_dataset import I2GDataset
except (ImportError, ModuleNotFoundError):
    I2GDataset = None

try:
    from .iid_dataset import IIDDataset
except (ImportError, ModuleNotFoundError):
    IIDDataset = None

try:
    from .ff_blend import FFBlendDataset
except (ImportError, ModuleNotFoundError):
    FFBlendDataset = None

try:
    from .fwa_blend import FWABlendDataset
except (ImportError, ModuleNotFoundError):
    FWABlendDataset = None

try:
    from .lrl_dataset import LRLDataset
except (ImportError, ModuleNotFoundError):
    LRLDataset = None

try:
    from .pair_dataset import pairDataset
except (ImportError, ModuleNotFoundError):
    pairDataset = None

try:
    from .sbi_dataset import SBIDataset
except (ImportError, ModuleNotFoundError):
    SBIDataset = None

try:
    from .lsda_dataset import LSDADataset
except (ImportError, ModuleNotFoundError):
    LSDADataset = None

try:
    from .tall_dataset import TALLDataset
except (ImportError, ModuleNotFoundError):
    TALLDataset = None
