from dataclasses import dataclass, field
from typing import Callable, Any
from torch.utils.data import ConcatDataset, Dataset
from uni2ts.data.builder._base import DatasetBuilder
from uni2ts.data.builder.simple import SimpleDatasetBuilder
from uni2ts.transform import Transformation


@dataclass
class MixedDatasetBuilder(DatasetBuilder):
    datasets: list
    storage_path: str
    weight: float = 1.0

    def build_dataset(self):
        raise NotImplementedError

    def load_dataset(self, transform_map: dict[Any, Callable[..., Transformation]]) -> Dataset:
        builders = [
            SimpleDatasetBuilder(
                dataset=name,
                weight=1.0,
                storage_path=self.storage_path,
            )
            for name in self.datasets
        ]
        return ConcatDataset([b.load_dataset(transform_map) for b in builders])
