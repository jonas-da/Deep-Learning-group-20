#
DEFAULT_TITLE_SIZE = 30
DEFAULT_BODY_SIZE = 40
UNKNOWN_TITLE_VALUE = [0] * DEFAULT_TITLE_SIZE
UNKNOWN_BODY_VALUE = [0] * DEFAULT_BODY_SIZE

DEFAULT_DOCUMENT_SIZE = 768

def print_hparams(hparams_class):
    for attr, value in hparams_class.__annotations__.items():
        # Print attribute names and values
        print(f"{attr}: {getattr(hparams_class, attr)}")


def hparams_to_dict(hparams_class) -> dict:
    params = {}
    for attr, value in hparams_class.__annotations__.items():
        params[attr] = getattr(hparams_class, attr)
    return params

class hparams_nrms:
    # Seed
    seed: int = 1
    # Dataset
    data_split: str = "ebnerd_small"
    train_fraction: float = 1.0
    np_ratio: int = 4
    test_fraction: float = 1
    # INPUT DIMENTIONS:
    num_classes: int = 5
    title_size: int = DEFAULT_TITLE_SIZE
    history_size: int = 20
    # MODEL ARCHITECTURE
    head_num: int = 20
    head_dim: int = 20
    word_embedding_dim:int = 768
    attention_hidden_dim: int = 200
    # MODEL OPTIMIZER:
    num_epoch: int = 5
    patience_counter: int = 5
    val_interval: int = 100
    train_batch_size: int = 32
    test_batch_size: int = 16
    dropout: float = 0.2
    learning_rate: float = 1e-4

class hparams_nrms_docvec:
    # Seed
    seed: int = 1
    # Dataset
    data_split: str = "ebnerd_small"
    train_fraction: float = 1
    np_ratio: int = 4
    test_fraction: float = 1
    # INPUT DIMENTIONS:
    num_classes: int = 5
    history_size: int = 20
    title_size: int = DEFAULT_DOCUMENT_SIZE
    # MODEL ARCHITECTURE
    head_num: int = 16
    head_dim: int = 16
    attention_hidden_dim: int = 200
    # MODEL OPTIMIZER:
    num_epoch: int = 5
    patience_counter: int = 5
    val_interval: int = 50
    train_batch_size: int = 32
    test_batch_size: int = 16
    dropout: float = 0.2
    learning_rate: float = 1e-4
    newsencoder_units_per_layer: list[int] = [512, 512, 512]
    newsencoder_l2_regularization: float = 1e-4
