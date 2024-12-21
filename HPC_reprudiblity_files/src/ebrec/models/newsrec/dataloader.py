from dataclasses import dataclass, field
import tensorflow as tf
import polars as pl
import numpy as np

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
class NewsrecDataLoader(tf.keras.utils.Sequence):
    """
    A DataLoader for news recommendation.
    """

    behaviors: pl.DataFrame
    history_column: str
    article_dict: dict[int, any]
    unknown_representation: str
    eval_mode: bool = False
    batch_size: int = 32
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL
    labels_col: str = DEFAULT_LABELS_COL
    user_col: str = DEFAULT_USER_COL
    kwargs: field(default_factory=dict) = None

    def __post_init__(self):
        """
        Post-initialization method. Loads the data and sets additional attributes.
        """
        self.lookup_article_index, self.lookup_article_matrix = create_lookup_objects(
            self.article_dict, unknown_representation=self.unknown_representation
        )
        self.unknown_index = [0]
        self.X, self.y = self.load_data()
        if self.kwargs is not None:
            self.set_kwargs(self.kwargs)

    def __len__(self) -> int:
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self):
        raise ValueError("Function '__getitem__' needs to be implemented.")

    def load_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        X = self.behaviors.drop(self.labels_col).with_columns(
            pl.col(self.inview_col).list.len().alias("n_samples")
        )
        y = self.behaviors[self.labels_col]
        return X, y

    def set_kwargs(self, kwargs: dict):
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class NRMSDataLoader(NewsrecDataLoader):
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
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

    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        """
        his_input_title:    (samples, history_size, document_dimension)
        pred_input_title:   (samples, npratio, document_dimension)
        batch_y:            (samples, npratio)
        """
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size].pipe(
            self.transform
        )
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        # =>
        if self.eval_mode:
            repeats = np.array(batch_X["n_samples"])
            # =>
            batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)
            # =>
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
            # =>
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].explode().to_list()
            ]
        else:
            batch_y = np.array(batch_y.to_list())
            his_input_title = self.lookup_article_matrix[
                batch_X[self.history_column].to_list()
            ]
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].to_list()
            ]
            pred_input_title = np.squeeze(pred_input_title, axis=2)

        his_input_title = np.squeeze(his_input_title, axis=2)
        return (his_input_title, pred_input_title), batch_y



@dataclass
class NRMSTemporalLayerDataLoader(NewsrecDataLoader):
    """DataLoader for NRMS model with temporal features.
    
    This dataloader handles both the article content and temporal features,
    ensuring proper shape and normalization of time-based signals.
    
    Attributes:
        behaviors (pl.DataFrame): DataFrame containing user behaviors
        history_column (str): Name of column containing user history
        article_dict (dict): Dictionary mapping article IDs to their embeddings
        unknown_representation (str): How to handle unknown articles
        eval_mode (bool): Whether in evaluation mode
        batch_size (int): Size of batches
        inview_col (str): Column name for candidate articles
        labels_col (str): Column name for labels
        user_col (str): Column name for user IDs
    """
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform article IDs to their corresponding embeddings."""
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

    def normalize_time_deltas(self, time_deltas: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Normalize time deltas and ensure correct shape.
        
        Args:
            time_deltas: Array of time differences in seconds
            eval_mode: Whether in evaluation mode (affects reshaping)
            
        Returns:
            Normalized time deltas with proper shape
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        
        # Replace None/NaN values with maximum time delta
        

        max_delta = np.nanmax(time_deltas) + epsilon
        
        
        time_deltas = np.nan_to_num(time_deltas, nan=max_delta)
        
        # Normalize to [0, 1] range using log-scale normalization
        # Adding 1 to avoid log(0) and to make very recent items close to 0
        normalized = np.log1p(time_deltas) / np.log1p(max_delta)
        
        # Shape handling
        if normalized.ndim == 1:
            normalized = normalized.reshape(-1, 1)
        
        return normalized
    
    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        """Get a batch of data.
        
        Args:
            idx: Batch index
            
        Returns:
            Tuple containing:
                - his_input_title: User history article embeddings
                - pred_input_title: Candidate article embeddings
                - time_deltas: Normalized time differences
                - batch_y: Labels
        """
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size].pipe(
            self.transform
        )
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        if self.eval_mode:
            # Evaluation mode - process all candidates
            repeats = np.array(batch_X["n_samples"])
            batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)
            
            # Process history
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
            
            # Process candidates
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].explode().to_list()
            ]
            
            # Process time deltas
            time_deltas = np.array(batch_X["time_delta"].explode().to_list())
#             non_none_values = [value for value in time_deltas if value is not None]
#             mean_value = sum(non_none_values) / len(non_none_values)

# # Replace None values with the mean
#             time_deltas = [mean_value if value is None else value for value in time_deltas]
            try:
                time_deltas = self.normalize_time_deltas(time_deltas, eval_mode=True)
            except:
                time_deltas = np.array(batch_X["time_delta"].explode().to_list())
                non_none_values = [value for value in time_deltas if value is not None]
                mean_value = sum(non_none_values) / len(non_none_values)

    # Replace None values with the mean
                time_deltas = [mean_value if value is None else value for value in time_deltas]
                # print(idx)
                # print(time_deltas)
                time_deltas = self.normalize_time_deltas(time_deltas, eval_mode=True)
            
        else:
            # Training mode - process fixed number of candidates
            batch_y = np.array(batch_y.to_list())
            
            # Process history
            his_input_title = self.lookup_article_matrix[
                batch_X[self.history_column].to_list()
            ]
            
            # Process candidates
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].to_list()
            ]
            pred_input_title = np.squeeze(pred_input_title, axis=2)
            
            # Process time deltas
            time_deltas = np.array(batch_X["time_delta"].to_list())
            time_deltas = self.normalize_time_deltas(time_deltas)

        # Final shape adjustments
        his_input_title = np.squeeze(his_input_title, axis=2)
        
        # Ensure time_deltas matches pred_input_title shape for training mode
        if not self.eval_mode:
            # Reshape time_deltas to match pred_input_title: (batch_size, n_candidates, 1)
            time_deltas = time_deltas.reshape(pred_input_title.shape[0], -1, 1)
        else:
            # For eval mode, maintain the proper shape based on all candidates
            time_deltas = time_deltas.reshape(-1, 1, 1)
        
        return (his_input_title, pred_input_title, time_deltas), batch_y



@dataclass
class NRMSTemporalDataLoader_ext(NewsrecDataLoader):
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
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

    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        """
        his_input_title:    (samples, history_size, document_dimension)
        pred_input_title:   (samples, npratio, document_dimension)
        discount_time_delta: (samples, npratio, document_dimension)
        batch_y:            (samples, npratio)
        """
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size].pipe(
            self.transform
        )
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        # =>
        if self.eval_mode:
            repeats = np.array(batch_X["n_samples"])
            # =>
            batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)
            # =>
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
            # =>
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].explode().to_list()
            ]

            discount_time_delta = np.array(batch_X["discount_time_delta"].explode().to_list()).reshape(-1, 1, 1)
        else:
            batch_y = np.array(batch_y.to_list())
            his_input_title = self.lookup_article_matrix[
                batch_X[self.history_column].to_list()
            ]
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].to_list()
            ]
            pred_input_title = np.squeeze(pred_input_title, axis=2)

            discount_time_delta = np.array(batch_X["discount_time_delta"].to_list())

        his_input_title = np.squeeze(his_input_title, axis=2)
        return (his_input_title, pred_input_title, discount_time_delta), batch_y

@dataclass
class NRMSTemporalDataLoader_Layer(NewsrecDataLoader):
    """DataLoader for NRMS model with temporal features.
    
    This dataloader handles both the article content and temporal features,
    ensuring proper shape and normalization of time-based signals.
    
    Attributes:
        behaviors (pl.DataFrame): DataFrame containing user behaviors
        history_column (str): Name of column containing user history
        article_dict (dict): Dictionary mapping article IDs to their embeddings
        unknown_representation (str): How to handle unknown articles
        eval_mode (bool): Whether in evaluation mode
        batch_size (int): Size of batches
        inview_col (str): Column name for candidate articles
        labels_col (str): Column name for labels
        user_col (str): Column name for user IDs
    """
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform article IDs to their corresponding embeddings."""
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

    def normalize_time_deltas(self, time_deltas: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Normalize time deltas and ensure correct shape.
        
        Args:
            time_deltas: Array of time differences in seconds
            eval_mode: Whether in evaluation mode (affects reshaping)
            
        Returns:
            Normalized time deltas with proper shape
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        
        # Replace None/NaN values with maximum time delta
        max_delta = np.nanmax(time_deltas) + epsilon
        time_deltas = np.nan_to_num(time_deltas, nan=max_delta)
        
        # Normalize to [0, 1] range using log-scale normalization
        # Adding 1 to avoid log(0) and to make very recent items close to 0
        normalized = np.log1p(time_deltas) / np.log1p(max_delta)
        
        # Shape handling
        if normalized.ndim == 1:
            normalized = normalized.reshape(-1, 1)
        
        return normalized
    
    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        """Get a batch of data.
        
        Args:
            idx: Batch index
            
        Returns:
            Tuple containing:
                - his_input_title: User history article embeddings
                - pred_input_title: Candidate article embeddings
                - time_deltas: Normalized time differences
                - batch_y: Labels
        """
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size].pipe(
            self.transform
        )
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        if self.eval_mode:
            # Evaluation mode - process all candidates
            repeats = np.array(batch_X["n_samples"])
            batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)
            
            # Process history
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
            
            # Process candidates
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].explode().to_list()
            ]
            
            # Process time deltas
            time_deltas = np.array(batch_X["time_delta"].explode().to_list())
            time_deltas = self.normalize_time_deltas(time_deltas, eval_mode=True)
            
        else:
            # Training mode - process fixed number of candidates
            batch_y = np.array(batch_y.to_list())
            
            # Process history
            his_input_title = self.lookup_article_matrix[
                batch_X[self.history_column].to_list()
            ]
            
            # Process candidates
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].to_list()
            ]
            pred_input_title = np.squeeze(pred_input_title, axis=2)
            
            # Process time deltas
            time_deltas = np.array(batch_X["time_delta"].to_list())
            time_deltas = self.normalize_time_deltas(time_deltas)

        # Final shape adjustments
        his_input_title = np.squeeze(his_input_title, axis=2)
        
        # Ensure time_deltas matches pred_input_title shape for training mode
        if not self.eval_mode:
            # Reshape time_deltas to match pred_input_title: (batch_size, n_candidates, 1)
            time_deltas = time_deltas.reshape(pred_input_title.shape[0], -1, 1)
        else:
            # For eval mode, maintain the proper shape based on all candidates
            time_deltas = time_deltas.reshape(-1, 1, 1)
        
        return (his_input_title, pred_input_title, time_deltas), batch_y



@dataclass
class NRMSDataLoaderPretransform(NewsrecDataLoader):
    """
    In the __post_init__ pre-transform the entire DataFrame. This is useful for
    when data can fit in memory, as it will be much faster ones training.
    Note, it might not be as scaleable.
    """

    def __post_init__(self):
        super().__post_init__()
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

    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        """
        his_input_title:    (samples, history_size, document_dimension)
        pred_input_title:   (samples, npratio, document_dimension)
        batch_y:            (samples, npratio)
        """
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        # =>
        if self.eval_mode:
            repeats = np.array(batch_X["n_samples"])
            # =>
            batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)
            # =>
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
            # =>
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].explode().to_list()
            ]
        else:
            batch_y = np.array(batch_y.to_list())
            his_input_title = self.lookup_article_matrix[
                batch_X[self.history_column].to_list()
            ]
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].to_list()
            ]
            pred_input_title = np.squeeze(pred_input_title, axis=2)

        his_input_title = np.squeeze(his_input_title, axis=2)
        return (his_input_title, pred_input_title), batch_y


@dataclass(kw_only=True)
class LSTURDataLoader(NewsrecDataLoader):
    """
    NPA and LSTUR shares the same DataLoader
    """

    user_id_mapping: dict[int, int] = None
    unknown_user_value: int = 0

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.pipe(
                map_list_article_id_to_value,
                behaviors_column=self.history_column,
                mapping=self.lookup_article_index,
                fill_nulls=self.unknown_index,
                drop_nulls=False,
            )
            .pipe(
                map_list_article_id_to_value,
                behaviors_column=self.inview_col,
                mapping=self.lookup_article_index,
                fill_nulls=self.unknown_index,
                drop_nulls=False,
            )
            .with_columns(
                pl.col(self.user_col).replace(
                    self.user_id_mapping, default=self.unknown_user_value
                )
            )
        )

    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        """
        user_indexes:       ()
        his_input_title:    (samples, history_size, document_dimension)
        pred_input_title:   (samples, npratio, document_dimension)
        batch_y:            (samples, npratio)
        """
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size].pipe(
            self.transform
        )
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        # =>
        if self.eval_mode:
            repeats = np.array(batch_X["n_samples"])
            # =>
            batch_y = np.array(batch_y.explode().to_list()).reshape(-1, 1)
            # =>
            user_indexes = np.array(
                batch_X.select(
                    pl.col(self.user_col).repeat_by(pl.col("n_samples")).explode()
                )[self.user_col].to_list()
            ).reshape(-1, 1)
            # =>
            his_input_title = repeat_by_list_values_from_matrix(
                batch_X[self.history_column].to_list(),
                matrix=self.lookup_article_matrix,
                repeats=repeats,
            )
            # =>
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].explode().to_list()
            ]
        else:
            # =>
            batch_y = np.array(batch_y.to_list())
            # =>
            user_indexes = np.array(batch_X[self.user_col].to_list()).reshape(-1, 1)
            # =>
            his_input_title = self.lookup_article_matrix[
                batch_X[self.history_column].to_list()
            ]
            # =>
            pred_input_title = self.lookup_article_matrix[
                batch_X[self.inview_col].to_list()
            ]
            pred_input_title = np.squeeze(pred_input_title, axis=2)
        # =>
        his_input_title = np.squeeze(his_input_title, axis=2)
        return (user_indexes, his_input_title, pred_input_title), batch_y


@dataclass(kw_only=True)
class NAMLDataLoader(NewsrecDataLoader):
    """
    Eval mode not implemented
    """

    unknown_category_value: int = 0
    unknown_subcategory_value: int = 0
    body_mapping: dict[int, list[int]] = None
    category_mapping: dict[int, int] = None
    subcategory_mapping: dict[int, int] = None

    def __post_init__(self):
        self.title_prefix = "title_"
        self.body_prefix = "body_"
        self.category_prefix = "category_"
        self.subcategory_prefix = "subcategory_"
        (
            self.lookup_article_index_body,
            self.lookup_article_matrix_body,
        ) = create_lookup_objects(
            self.body_mapping, unknown_representation=self.unknown_representation
        )
        if self.eval_mode:
            raise ValueError("'eval_mode = True' is not implemented for NAML")

        return super().__post_init__()

    def transform(self, df: pl.DataFrame) -> tuple[pl.DataFrame]:
        """
        Special case for NAML as it requires body-encoding, verticals, & subvertivals
        """
        # =>
        title = df.pipe(
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
        # =>
        body = df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.lookup_article_index_body,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.lookup_article_index_body,
            fill_nulls=self.unknown_index,
            drop_nulls=False,
        )
        # =>
        category = df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.category_mapping,
            fill_nulls=self.unknown_category_value,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.category_mapping,
            fill_nulls=self.unknown_category_value,
            drop_nulls=False,
        )
        # =>
        subcategory = df.pipe(
            map_list_article_id_to_value,
            behaviors_column=self.history_column,
            mapping=self.subcategory_mapping,
            fill_nulls=self.unknown_subcategory_value,
            drop_nulls=False,
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=self.inview_col,
            mapping=self.subcategory_mapping,
            fill_nulls=self.unknown_subcategory_value,
            drop_nulls=False,
        )
        return (
            pl.DataFrame()
            .with_columns(title.select(pl.all().name.prefix(self.title_prefix)))
            .with_columns(body.select(pl.all().name.prefix(self.body_prefix)))
            .with_columns(category.select(pl.all().name.prefix(self.category_prefix)))
            .with_columns(
                subcategory.select(pl.all().name.prefix(self.subcategory_prefix))
            )
        )

    def __getitem__(self, idx) -> tuple[tuple[np.ndarray], np.ndarray]:
        batch_X = self.X[idx * self.batch_size : (idx + 1) * self.batch_size].pipe(
            self.transform
        )
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        # =>
        batch_y = np.array(batch_y.to_list())
        his_input_title = np.array(
            batch_X[self.title_prefix + self.history_column].to_list()
        )
        his_input_body = np.array(
            batch_X[self.body_prefix + self.history_column].to_list()
        )
        his_input_vert = np.array(
            batch_X[self.category_prefix + self.history_column].to_list()
        )[:, :, np.newaxis]
        his_input_subvert = np.array(
            batch_X[self.subcategory_prefix + self.history_column].to_list()
        )[:, :, np.newaxis]
        # =>
        pred_input_title = np.array(
            batch_X[self.title_prefix + self.inview_col].to_list()
        )
        pred_input_body = np.array(
            batch_X[self.body_prefix + self.inview_col].to_list()
        )
        pred_input_vert = np.array(
            batch_X[self.category_prefix + self.inview_col].to_list()
        )[:, :, np.newaxis]
        pred_input_subvert = np.array(
            batch_X[self.subcategory_prefix + self.inview_col].to_list()
        )[:, :, np.newaxis]
        # =>
        his_input_title = np.squeeze(
            self.lookup_article_matrix[his_input_title], axis=2
        )
        pred_input_title = np.squeeze(
            self.lookup_article_matrix[pred_input_title], axis=2
        )
        his_input_body = np.squeeze(
            self.lookup_article_matrix_body[his_input_body], axis=2
        )
        pred_input_body = np.squeeze(
            self.lookup_article_matrix_body[pred_input_body], axis=2
        )
        # =>
        return (
            his_input_title,
            his_input_body,
            his_input_vert,
            his_input_subvert,
            pred_input_title,
            pred_input_body,
            pred_input_vert,
            pred_input_subvert,
        ), batch_y
