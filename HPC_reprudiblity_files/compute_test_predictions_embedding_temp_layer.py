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
import shutil



from ebrec.utils._constants import *
from ebrec.evaluation.beyond_accuracy import (
    IntralistDiversity,
    Distribution,
    Serendipity,
    Sentiment,
    Coverage,
    Novelty,
)

from src.ebrec.models.newsrec.model_config import (
    hparams_nrms,
    hparams_nrms_docvec,
    hparams_to_dict,
    print_hparams,
)
from ebrec.utils._articles import create_sort_based_prediction_score
from ebrec.utils._behaviors import truncate_history
from ebrec.utils._polars import slice_join_dataframes
from ebrec.utils._python import (
    rank_predictions_by_score,
    write_submission_file,
    write_json_file,
    read_json_file,
)
from ebrec.utils._polars import split_df_chunks, concat_str_columns
from ebrec.utils._articles import create_article_id_to_value_mapping

from ebrec.utils._constants import *
from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_known_user_column,
    add_prediction_scores,
    truncate_history,
    # ebnerd_from_path,
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

ARTICLES_PATH = PATH.joinpath("ebnerd_large/articles.parquet")

DUMP_DIR_BASELINES = DUMP_DIR.joinpath("baselines")
DUMP_DIR_BASELINES.mkdir(parents=True, exist_ok=True)


MODEL_NAME = f"NRMS-{DT_NOW}"


BEYOND_ACCURACY_HISTORY_DICT = "beyond_accuracy_history_dict.json"
BEYOND_ACCURACY_USERS_DICT = "beyond_accuracy_users_dict.json"
CANDIDATE_LIST = "candidate_list.json"
ARTICLES_DICT = "articles_dict.json"
BEHAVIORS_TIMESTAMP_DICT = "behaviors_timestamp_dict.json"
#
BASELINE_DIVERSITY = "intralist_diversity.json"
BASELINE_SENTIMENT_SCORE = "sentiment_score.json"
BASELINE_NOVELTY = "novelty.json"
BASELINE_SERENDIPITY = "serendipity.json"
BASELINE_COVERAGE = "coverage.json"
BASELINE_DISTRIBUTION_CATEGORY = "distribution_category.json"
BASELINE_DISTRIBUTION_SENTIMENT_LABEL = "distribution_sentiment_label.json"
BASELINE_DISTRIBUTION_TOPICS = "distribution_topics.json"   
# Create the folder, including any intermediate directories
model_name = 'temp_layer'

BATCH_SIZE = 32
learning_rate = 1e-4
model_name = 'Temp_lay'
embedding = 'xlm_roberta_base'
HISTORY_SIZE = 35

MODEL_WEIGHTS = f"./runs/state_dict/temporal_tuning/"+str(BATCH_SIZE)+"_"+str(learning_rate)+"_"+str(embedding)+"_"+str(HISTORY_SIZE)+"_"+model_name+"/weights"
Path(MODEL_WEIGHTS).mkdir(parents=True, exist_ok=True)
LOG_DIR = DUMP_DIR.joinpath(f"./runs/{MODEL_NAME}")
EPOCHS = 25


BATCH_SIZE_TEST_WO_B = 32
BATCH_SIZE_TEST_W_B = 32

DUMP_DIR = Path("ebnerd_predictions")
DUMP_DIR.mkdir(exist_ok=True, parents=True)
DT_NOW = dt.datetime.now()
MODEL_OUTPUT_NAME = f"{MODEL_NAME}-{DT_NOW}"
ARTIFACT_DIR = DUMP_DIR.joinpath("test_predictions", MODEL_OUTPUT_NAME)
LOG_DIR = DUMP_DIR.joinpath(f"runs/{MODEL_OUTPUT_NAME}")
TEST_CHUNKS_DIR = ARTIFACT_DIR.joinpath("test_chunks")
TEST_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

portion = int(sys.argv[1])

COLUMNS = [
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_USER_COL,
]

user_meta_columns = [
    DEFAULT_IS_SUBSCRIBER_COL,
    DEFAULT_IS_SSO_USER_COL,
    DEFAULT_POSTCODE_COL,
    DEFAULT_GENDER_COL,
    DEFAULT_AGE_COL,
]

df_articles = pl.read_parquet(PATH.joinpath("ebnerd_small/articles.parquet"))

precomputed_embeddings = pl.read_parquet(PATH.joinpath("embeddings/"+embedding+".parquet"))

precomputed_embeddings = precomputed_embeddings.filter(precomputed_embeddings['article_id'].is_in(df_articles['article_id']))

pre_embs = np.array([precomputed_embeddings['embeddings'][0]])
article_mapping = create_article_id_to_value_mapping(
    df=precomputed_embeddings,
    value_col="embeddings",  # Column containing precomputed embeddings
    article_col="article_id",  # Column containing article IDs
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

gc.collect()

print("loading model...")
model.model.load_weights(MODEL_WEIGHTS)


df_test_glob = (
    ebnerd_from_path(PATH.joinpath("ebnerd_testset", "test"), history_size=HISTORY_SIZE)
    .sample(fraction=1)
    .with_columns(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL)
        .list.first()
        .alias(DEFAULT_CLICKED_ARTICLES_COL)
    )
    .select(COLUMNS + [DEFAULT_IS_BEYOND_ACCURACY_COL])
    .with_columns(
        pl.col(DEFAULT_INVIEW_ARTICLES_COL)
        .list.eval(pl.element() * 0)
        .alias(DEFAULT_LABELS_COL)
    )
)

# Create article time dictionary
article_time_dict = create_article_time_dict(df_articles)
# Add temporal features to your datasets
df_test_glob = prepare_temporal_features(
    df_test_glob,
    article_time_dict,
    DEFAULT_INVIEW_ARTICLES_COL
)

# split the dataset in 5 parts, as when running for the whole dataset it crashed on the HPC
n = len(df_test_glob)
splits = 5
quarter = n // splits
df_tests = []
for i in range(splits):
    
    if i == splits-1:
        df_tests.append(df_test_glob[(i) * quarter :])
    elif i == 0:
        df_tests.append(df_test_glob[:quarter])
    else:
        df_tests.append(df_test_glob[i * quarter : (i + 1) * quarter])

df_test = df_tests[portion-1]

# Split test in beyond-accuracy TRUE / FALSE. In the BA 'article_ids_inview' is 250.
if portion == splits:
    
    df_test_wo_beyond = df_test.filter(~pl.col(DEFAULT_IS_BEYOND_ACCURACY_COL))
    df_test_w_beyond = df_test.filter(pl.col(DEFAULT_IS_BEYOND_ACCURACY_COL))

else:
    df_test_wo_beyond = df_test

df_test_chunks = df_test_wo_beyond

print("Initiating testset without beyond-accuracy...")

test_dataloader_wo_b = NRMSTemporalLayerDataLoader(
    behaviors=df_test_chunks,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=BATCH_SIZE_TEST_WO_B,
)

# Predict and clear session
scores = model.scorer.predict(test_dataloader_wo_b)
tf.keras.backend.clear_session()

# Process the predictions
df_test_chunks = add_prediction_scores(df_test_chunks, scores.tolist()).with_columns(
    pl.col("scores")
    .map_elements(lambda x: list(rank_predictions_by_score(x)))
    .alias("ranked_scores")
)

# Save the processed chunk
df_test_chunks.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    TEST_CHUNKS_DIR.joinpath(f"pred_wo_ba_{i}.parquet")
)

# Cleanup
del test_dataloader_wo_b, scores
gc.collect()

df_pred_test_wo_beyond = df_test_chunks
df_pred_test_wo_beyond.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    TEST_CHUNKS_DIR.joinpath("pred_wo_ba.parquet")
)
# =====================================================================================
if portion == splits:
        
    print("Initiating testset with beyond-accuracy...")
    test_dataloader_w_b = NRMSTemporalLayerDataLoader(
    behaviors=df_test_w_beyond,
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=BATCH_SIZE_TEST_W_B,
)
    
    scores = model.scorer.predict(test_dataloader_w_b)
    df_pred_test_w_beyond = add_prediction_scores(
        df_test_w_beyond, scores.tolist()
    ).with_columns(
        pl.col("scores")
        .map_elements(lambda x: list(rank_predictions_by_score(x)))
        .alias("ranked_scores")
    )
    df_pred_test_w_beyond.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
        TEST_CHUNKS_DIR.joinpath("pred_w_ba.parquet")
    )
    df_test = pl.concat([df_pred_test_wo_beyond, df_pred_test_w_beyond])

else:
    df_test = df_pred_test_wo_beyond

# =====================================================================================
print("Saving prediction results...")

df_test.select(DEFAULT_IMPRESSION_ID_COL, "ranked_scores").write_parquet(
    ARTIFACT_DIR.joinpath("test_predictions.parquet")
)

if TEST_CHUNKS_DIR.exists() and TEST_CHUNKS_DIR.is_dir():
    shutil.rmtree(TEST_CHUNKS_DIR)

path_sub = "runs/prediction_output/sub_file/"+str(BATCH_SIZE)+"_"+str(learning_rate)+"_"+str(embedding)+"_"+str(HISTORY_SIZE)+"_"+str(model_name)
Path(path_sub).mkdir(parents=True, exist_ok=True)
write_submission_file(
    impression_ids=df_test[DEFAULT_IMPRESSION_ID_COL],
    prediction_scores=df_test["ranked_scores"],
    path=path_sub+"/predictions_"+str(portion)+".txt",
    filename_zip=path_sub+"/predictions_"+str(portion)+".zip",
)

print("Correctly ended")

