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
    A DataLoader for news recommendation.

    Attributes:
        behaviors (pl.DataFrame): The user behaviors data.
        history_column (str): Column name for user reading history.
        article_dict (dict[int, any]): A dictionary mapping article IDs to metadata.
        unknown_representation (str): Placeholder for unknown article representations.
        eval_mode (bool): Indicates if the DataLoader is in evaluation mode.
        inview_col (str): Column name for currently viewed articles.
        labels_col (str): Column name for labels (e.g., clicks).
        user_col (str): Column name for user identifiers.
        kwargs (dict): Additional parameters for customization.
    
    Example:
        behaviors = pl.DataFrame({"history": [[1, 2]], "inview": [[3, 4]], "labels": [1]})
        dataloader = NewsrecDataLoader(
            behaviors=behaviors,
            history_column="history",
            article_dict={1: "Article A", 2: "Article B"},
            unknown_representation="<UNK>",
        )
    """
    behaviors: pl.DataFrame
    history_column: str
    article_dict: dict[int, any]
    unknown_representation: str
    eval_mode: bool = False
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL
    labels_col: str = DEFAULT_LABELS_COL
    user_col: str = DEFAULT_USER_COL
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        """
        Post-initialization method. Loads the data and sets additional attributes.
        """
        # Create lookup objects for article mappings
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation
        )
        self.unknown_index = [0]  # Placeholder index for unknown articles
        
        # Load features (X) and labels (y)
        self.X, self.y = self.load_data()
        
        # Apply additional customization parameters if provided
        if self.kwargs is not None:
            self.set_kwargs(self.kwargs)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Length of the dataset.
        """
        return int(len(self.X))

    def __getitem__(self):
        """
        Placeholder for item fetching. Needs implementation in subclasses.

        Raises:
            ValueError: If the method is not implemented.
        """
        raise ValueError("Function '__getitem__' needs to be implemented.")

    def load_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Prepares the features (X) and labels (y) from the behaviors DataFrame.

        Returns:
            tuple[pl.DataFrame, pl.DataFrame]: Features and labels DataFrames.
        """
        X = self.behaviors.drop(self.labels_col).with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")  # Calculate sample count per row
        )
        y = self.behaviors[self.labels_col]  # Extract labels
        return X, y

    def set_kwargs(self, kwargs: dict):
        """
        Sets additional attributes dynamically from a kwargs dictionary.

        Args:
            kwargs (dict): Key-value pairs of attributes to set.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

@dataclass
class NRMSDataLoaderPretransform(NewsrecDataLoader):
    """
    A unified DataLoader for training and validation/testing in news recommendation.

    Extends:
        NewsrecDataLoader

    Attributes:
        eval (bool): Indicates evaluation mode (True for validation/test).
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
        """
        Extends the post-initialization method to transform features.
        """
        super().__post_init__()

        # Map article IDs to their respective indices for history and inview columns
        self.X = self.X.pipe(
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

    def __getitem__(self, idx):
        """
        Fetches a single data sample for training or evaluation.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing feature tensors and label tensors.
        """
        X = self.X[idx]
        y = np.array(self.y[idx])  # Convert labels to numpy array

        if not self.eval:
            # Training logic: Prepare input tensors for history and predictions
            his_input_title = self.lookup_article_matrix[np.array(X[self.history_column].to_list())]
            pred_input_title = self.lookup_article_matrix[np.array(X[self.inview_col].to_list())]
            his_input_title = np.squeeze(his_input_title, axis=(0, 2))
            pred_input_title = np.squeeze(pred_input_title, axis=(0, 2))
        else:
            # Validation/Testing logic: Repeat input for multiple samples
            repeats = np.array(X["n_samples"])
            his_input_title = repeat_by_list_values_from_matrix(
                X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
            his_input_title = np.squeeze(his_input_title, axis=2)
            pred_input_title = self.lookup_article_matrix[X[self.inview_col].explode().to_list()]

        return (torch.tensor(his_input_title, dtype=torch.float),
                torch.tensor(pred_input_title, dtype=torch.float)), torch.tensor(y, dtype=torch.float)
    

def nrms_custom_collate_fn(batch):
    """
    Custom collate function for batching data in the NRMS model.

    Args:
        batch (list): List of samples fetched by the DataLoader.

    Returns:
        tuple: A tuple containing batched features and labels.
    """
    # Unpack batch into separate lists
    his_input_titles, pred_input_title, y = zip(*[(item[0][0], item[0][1], item[1]) for item in batch])

    # Concatenate tensors directly
    his_input_titles_concatenated = torch.cat(his_input_titles, dim=0)
    pred_input_title_concatenated = torch.cat(pred_input_title, dim=0)
    y_concatenated = torch.cat(y, dim=0).unsqueeze(1)  # Add an extra axis for shape (sum, 1)

    return (his_input_titles_concatenated, pred_input_title_concatenated), y_concatenated