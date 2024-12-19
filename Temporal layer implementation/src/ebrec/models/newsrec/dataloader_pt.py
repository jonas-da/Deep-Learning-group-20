from dataclasses import dataclass, field
import torch
import numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader

from ebrec.utils._articles_behaviors import map_list_article_id_to_value
from ebrec.utils._python import (
    repeat_by_list_values_from_matrix,
    create_lookup_objects,
)

from ebrec.utils._constants import (
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_USER_COL,
)

@dataclass
class NewsrecDataLoader(Dataset):
    """
    A unified DataLoader for training and validation/testing in news recommendation.
    """

    behaviors: pl.DataFrame
    history_column: str
    article_dict: dict[int, any]
    unknown_representation: str
    eval: bool = False  # Set to True for validation/test mode
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL
    labels_col: str = DEFAULT_LABELS_COL
    user_col: str = DEFAULT_USER_COL
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation
        )
        self.unknown_index = [0]
        self.X, self.y = self.load_data()
        if self.kwargs is not None:
            self.set_kwargs(self.kwargs)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx].pipe(self.transform)

        y = np.array(self.y[idx])  # Convert labels to numpy array

        if not self.eval:
            # Training logic
            his_input_title = self.lookup_article_matrix[np.array(X[self.history_column].to_list())]
            pred_input_title = self.lookup_article_matrix[np.array(X[self.inview_col].to_list())]
            his_input_title = np.squeeze(his_input_title, axis=(0, 2))
            pred_input_title = np.squeeze(pred_input_title, axis=(0, 2))
        else:
            # Validation/Testing logic
            repeats = np.array(X["n_samples"])
            his_input_title = repeat_by_list_values_from_matrix(
                X[self.history_column].to_list(), 
                matrix=self.lookup_article_matrix, 
                repeats=repeats,
            )
            his_input_title = np.squeeze(his_input_title, axis=2)
            pred_input_title = self.lookup_article_matrix[X[self.inview_col].explode().to_list()]

        return (torch.tensor(his_input_title, dtype=torch.long),
                torch.tensor(pred_input_title, dtype=torch.long)), torch.tensor(y, dtype=torch.float)

    def transform(self, df: pl.DataFrame):
        return df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )

    def load_data(self):
        X = self.behaviors.drop(self.labels_col).with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")
        )
        y = self.behaviors[self.labels_col]
        return X, y

    def set_kwargs(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
