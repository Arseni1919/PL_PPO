from typing import Union, List, Any

from CONSTANTS import *
from alg_dataset import ALGDataset


class ALGDataModule(pl.LightningDataModule):

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.env = gym.make(ENV)
        self.dataset = ALGDataset(self.net, self.env)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage=None):
        # transforms
        pass

    def train_dataloader(self):
        return DataLoader(self.dataset)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass


