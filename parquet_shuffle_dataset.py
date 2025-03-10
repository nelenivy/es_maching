import pyarrow.parquet as pq
import logging
import pandas as pd
import torch
import numpy as np
from ptls.data_load.datasets import ParquetDataset, ParquetFiles
from ptls.data_load.iterable_processing.feature_filter import FeatureFilter
from ptls.data_load import IterableChain
from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset
from ptls.data_load.iterable_processing.to_torch_tensor import ToTorch

from functools import reduce
from operator import iadd
from typing import Union, List
from ptls.data_load.feature_dict import FeatureDict
from ptls.data_load.utils import collate_feature_dict
from ptls.data_load import read_pyarrow_file

logger = logging.getLogger(__name__)

class NoThreadsParquetDataset(ParquetDataset):
    """
        Parquet dataset without using multithreading reading which can cause crash whith multiprocessing

    """
    def __init__(self, data_files: Union[ParquetFiles, List[str]],
                 post_processing=None,
                 i_filters: List = None,
                 shuffle_files=False, cache_schema=True, shuffle_seed=42):
        super().__init__(data_files=data_files,
                 post_processing=post_processing,
                 i_filters=i_filters,
                 shuffle_files=shuffle_files, cache_schema=cache_schema, shuffle_seed=shuffle_seed)

    def iter_file(self, file_name):
        """

        :param file_name:
        :return: [(customer_id, features)]
        """
        logger.debug(f'[{self._worker_id}/{self._num_workers}] Iter file "{file_name}"')
        for rec in read_pyarrow_file(file_name, use_threads=False):
            rec = {k: self.to_torch(v) for k, v in rec.items()}
            yield rec

                
class ShuffleParquetDataset(NoThreadsParquetDataset):
    """Lazy (IterableDataset) load from parquet files

    File structure:
    *.parquet

    File structure example:
    data/
        part1.parquet
        part2.parquet
        ...

    Each file is read sequentially

    Parameters
    ----------
    data_files:
        ParquetFile object with list of files or just list of files
    post_processing:
        - deprecated, use i_filters
    i_filters:
        - list of `ptls.data_load.iterable_processing` filters
    shuffle_files:
        - shuffle data_files before reading when True.
    shuffle_one_file:
        - shuffle contents of one file.
    cache_schema:
        - dict schema (feature names) will be read once
    shuffle_seed:
        - random seed for shuffle_files

    """
    def __init__(self, data_files: Union[ParquetFiles, List[str]],
                 post_processing=None,
                 i_filters: List = None,
                 shuffle_files=False, shuffle_one_file=False, cache_schema=True, shuffle_seed=42):
        super().__init__(data_files=data_files,
                 post_processing=post_processing,
                 i_filters=i_filters,
                 shuffle_files=shuffle_files, cache_schema=cache_schema, shuffle_seed=shuffle_seed)
        self.shuffle_one_file=shuffle_one_file

    def iter_file(self, file_name):
        """

        :param file_name:
        :return: [(customer_id, features)]
        """
        logger.debug(f'[{self._worker_id}/{self._num_workers}] Iter file "{file_name}"')
        if self.shuffle_one_file:
            recs = []
            for rec in read_pyarrow_file(file_name, use_threads=False):
                recs.append({k: v for k, v in rec.items()})
            rs = np.random.RandomState(self._shuffle_seed % 2**32)
            rs.shuffle(recs)
            
            for rec in recs:
                rec_torch = {k: self.to_torch(v) for k, v in rec.items()}
                yield rec_torch
                
            del recs                
        else:
            for rec in super().iter_file(file_name):
                yield rec