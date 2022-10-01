from .main_data_module import *


def get_datamodule(dataset_name, **dataset_params) -> LightningDataModule:
    dataset_class = getattr(sys.modules[__name__], dataset_name)
    return MainDataModule(dataset_class, **dataset_params)
