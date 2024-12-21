from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf
import datetime as dt
import polars as pl
import gc
import os
from pathlib import Path
import sys
import numpy as np
import yaml


from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
)

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_known_user_column,
    add_prediction_scores,
    truncate_history,
)
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from ebrec.utils._articles import convert_text2encoding_with_transformers
from ebrec.utils._polars import concat_str_columns, slice_join_dataframes
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._python import write_submission_file, rank_predictions_by_score

from src.ebrec.models.newsrec.dataloader import NRMSDataLoader
from src.ebrec.models.newsrec.dataloader import NRMSTemporalDataLoader_ext
from src.ebrec.models.newsrec.dataloader import NRMSTemporalDataLoader_Layer
from src.ebrec.models.newsrec.model_config import hparams_nrms
from src.ebrec.models.newsrec import NRMSModel
from src.ebrec.models.newsrec.nrms_docvec import NRMSDocVec
from src.ebrec.models.newsrec.nrms_docvec_temp_simpl import NRMSDocVec_temp_simp
from src.ebrec.models.newsrec.nrms_docvec_temp_layer import NRMSDocVec_temp_layer
from src.ebrec.models.newsrec.dataloader import NRMSTemporalLayerDataLoader

from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime, timedelta

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tf.config.optimizer.set_jit(False)

# Prepare temporal features

def prepare_temporal_features(
    df: pl.DataFrame,
    article_time_dict: Dict[int, datetime],
    inview_col: str
) -> pl.DataFrame:
    """Add temporal features using vectorized operations."""
    # Add published times
    df = df.with_columns([
        pl.col(inview_col).map_elements(
            lambda ids: [article_time_dict.get(id) for id in ids],
            return_dtype=pl.List(pl.Datetime)
        ).alias(f"published_time_{inview_col}")
    ])
    # Add reference date (latest date from inview articles)
    df = df.with_columns(
        pl.col(f"published_time_{inview_col}")
        .map_elements(
            lambda dates: max((d for d in dates if d), default=None),
            return_dtype=pl.Datetime
        )
        .alias("reference_date")
    )
    # Calculate time differences in seconds
    df = df.with_columns([
        pl.struct([f"published_time_{inview_col}", "reference_date"])
        .map_elements(
            lambda row: calculate_time_difference_seconds(
                row[f"published_time_{inview_col}"], 
                row["reference_date"]
            ),
            return_dtype=pl.List(pl.Float64)
        ).alias("time_delta")
    ])
    return df
def calculate_time_difference_seconds(
    timestamps: List[Optional[datetime]], 
    reference_time: datetime
) -> List[Optional[float]]:
    """Calculate time differences in seconds between timestamps and reference time."""
    return [
        (reference_time - timestamp).total_seconds() 
        if timestamp else None 
        for timestamp in timestamps
    ]

def create_article_time_dict(df_articles: pl.DataFrame) -> Dict[int, datetime]:
    """Create lookup dictionary for article publishing times"""
    return dict(zip(
        df_articles["article_id"].to_list(),
        df_articles["published_time"].to_list()
    ))

def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.DataFrame:
    """
    Load ebnerd - function
    """
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
        )
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
        .pipe(
            slice_join_dataframes,
            df2=df_history.collect(),
            on=DEFAULT_USER_COL,
            how="left",
        )
    )
    return df_behaviors


PATH = Path("./data").expanduser()
DUMP_DIR = PATH.joinpath("ebnerd_predictions")
DUMP_DIR.mkdir(exist_ok=True, parents=True)
DT_NOW = dt.datetime.now()

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


DATASPLIT = "ebnerd_small"
FRACTION = 1

EPOCHS = 25

# values of the parameters over which we want to iterate
embeddings = ['bert_base_multilingual_cased', 'contrastive_vector', 'document_vector', 'xlm_roberta_base']
batch_size = [32, 64]
l_rates = [1e-6, 1e-5, 1e-4, 1e-3]

combs = []

for emb in embeddings:
    for bs in batch_size:
        for lr in l_rates:
            combs.append({'embedding': emb, 'batch size': bs, 'learning rate': lr, 'model': 'Temp_lay', 'history size': 35}) 


idx = int(sys.argv[1])

BATCH_SIZE = combs[idx-1]['batch size']
learning_rate = combs[idx-1]['learning rate']
embedding = combs[idx-1]['embedding']

model_name = 'Temp_lay'
HISTORY_SIZE =  35

MODEL_NAME = f"NRMS-{DT_NOW}"



# Create the folder, including any intermediate directories

MODEL_WEIGHTS = f"./runs/state_dict/temporal_tuning/"+str(BATCH_SIZE)+"_"+str(learning_rate)+"_"+str(embedding)+"_"+str(HISTORY_SIZE)+"_"+str(model_name)+"/weights"
Path(MODEL_WEIGHTS).mkdir(parents=True, exist_ok=True)
LOG_DIR = DUMP_DIR.joinpath(f"./runs/{MODEL_NAME}")

COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
]

df_train = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE)
    .sample(fraction=FRACTION)
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=4,
        shuffle=True,
        with_replacement=True,
        seed=123,
    )
    .pipe(create_binary_labels_column)
)

# =>
df_validation= (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "validation"), history_size=HISTORY_SIZE)
    .sample(fraction=FRACTION)
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=4,
        shuffle=True,
        with_replacement=True,
        seed=123,
    )
    .pipe(create_binary_labels_column)
)

df_test = pl.read_parquet('validation.parquet')

# Load articles embeddings
df_articles = pl.read_parquet(PATH.joinpath("ebnerd_small/articles.parquet"))

precomputed_embeddings = pl.read_parquet(PATH.joinpath("embeddings/"+embedding+".parquet"))

precomputed_embeddings = precomputed_embeddings.filter(precomputed_embeddings['article_id'].is_in(df_articles['article_id']))

pre_embs = np.array([precomputed_embeddings['embeddings'][0]])
article_mapping = create_article_id_to_value_mapping(
    df=precomputed_embeddings,
    value_col="embeddings",  # Column containing precomputed embeddings
    article_col="article_id",  # Column containing article IDs
)


# Create article time dictionary
article_time_dict = create_article_time_dict(df_articles)
# Add temporal features to your datasets
df_train = prepare_temporal_features(
    df_train,
    article_time_dict,
    DEFAULT_INVIEW_ARTICLES_COL
)
df_validation = prepare_temporal_features(
    df_validation,
    article_time_dict,
    DEFAULT_INVIEW_ARTICLES_COL
)
df_test = prepare_temporal_features(
    df_test,
    article_time_dict,
    DEFAULT_INVIEW_ARTICLES_COL
)
 

train_dataloader = NRMSTemporalLayerDataLoader(
    behaviors=df_train,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BATCH_SIZE,
)
val_dataloader = NRMSTemporalLayerDataLoader(
    behaviors=df_validation,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=BATCH_SIZE,
)

test_dataloader = NRMSTemporalLayerDataLoader(
    behaviors=df_test,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=BATCH_SIZE,
)

hparams_nrms.history_size = HISTORY_SIZE
hparams_nrms.learning_rate = learning_rate
hparams_nrms.title_size = pre_embs[0].shape[0]


model = NRMSDocVec_temp_layer(
    hparams=hparams_nrms,
    seed=42,
)

model.model.compile(
    optimizer=model.model.optimizer,
    loss=model.model.loss,
    metrics=["AUC"],
)


# CALLBACKS
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

# Earlystopping:
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="max",
    patience=3,
    restore_best_weights=True,
)

# ModelCheckpoint:
modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_WEIGHTS,
    monitor="val_loss",
    mode="max",
    save_best_only=False,
    save_weights_only=True,
    verbose=1,
)

# Learning rate scheduler:
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode="max",
    factor=0.2,
    patience=2,
    min_lr=learning_rate,
)
# modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(


hist = model.model.fit(
    train_dataloader,
    validation_data=val_dataloader,
    epochs=EPOCHS,
    callbacks=[tensorboard_callback, early_stopping, modelcheckpoint, lr_scheduler],
)


tr_loss = hist.history['loss']
val_loss = hist.history['val_loss']
losses = [tr_loss, val_loss]

pred_test = model.scorer.predict(test_dataloader)
df_test = add_prediction_scores(df_test, pred_test.tolist())

aucsc = AucScore()
auc = aucsc.calculate(y_true=df_test["labels"].to_list(), y_pred=df_test["scores"].to_list())


file_path = "runs/results/temporal_tuning/losses_"+str(BATCH_SIZE)+"_"+str(learning_rate)+"_"+str(embedding)+"_"+str(HISTORY_SIZE)+"_"+str(model_name)+".txt"
with open(file_path, 'w') as file:

    file.write(f"AUC on val: {auc}\n")
    for sublist in losses:
        file.write(f"{sublist}\n")

gc.collect()

print("saving model...")
model.model.save_weights(MODEL_WEIGHTS)


print("Correctly ended")