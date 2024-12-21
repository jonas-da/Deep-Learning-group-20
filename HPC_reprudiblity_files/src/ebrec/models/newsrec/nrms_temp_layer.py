# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from ebrec.models.newsrec.layers import AttLayer2, SelfAttention
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding, Input, Dropout, Dense, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2

class TemporalLayer(tf.keras.layers.Layer):
    """Custom layer to learn temporal relationships in news recommendations.
    This layer takes time differences as input and learns a temporal weighting function.
    Instead of using a fixed exponential decay, it allows the model to learn the optimal
    temporal weighting scheme.
    """
    def __init__(self, units=64, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = 0.2
    def build(self, input_shape):
        # Create trainable weights for temporal transformation
        self.temporal_transform = tf.keras.layers.Dense(
            self.units,
            activation=self.activation,
            kernel_initializer='glorot_uniform',
            name='temporal_transform'
        )
        # Final projection to scalar weight
        self.temporal_intermediate = tf.keras.layers.Dense(
            400, 
            activation=self.activation,  # Ensure output is between 0 and 1
            kernel_initializer='glorot_uniform',
            name='temporal_intermediate'
        )
        # Final projection to scalar weight
        self.temporal_intermediate_2 = tf.keras.layers.Dense(
            400, 
            activation=self.activation,  # Ensure output is between 0 and 1
            kernel_initializer='glorot_uniform',
            name='temporal_intermediate_2'
        )
        # Final projection to scalar weight
        self.temporal_weight = tf.keras.layers.Dense(
            1, 
            activation='sigmoid',  # Ensure output is between 0 and 1
            kernel_initializer='glorot_uniform',
            name='temporal_weight'
        )
        self.temporal_dropout = tf.keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)
    def call(self, inputs, training=None):
        # inputs shape: (batch_size, sequence_length, 1)
        # Transform temporal features through MLP
        x = self.temporal_transform(inputs)  # (batch_size, sequence_length, units)
        x = self.temporal_intermediate(x)
        x = self.temporal_dropout(x)
        x = self.temporal_intermediate_2(x)
        x = self.temporal_dropout(x)
        # Project to temporal weights
        temporal_weights = self.temporal_weight(x)  # (batch_size, sequence_length, 1)

        temporal_weights = tf.ensure_shape(temporal_weights, (None, None, 1))
        temporal_weights_400 = tf.tile(temporal_weights, [1, 1, 400])  # Shape: (batch_size, 400)
        temporal_weights_400 = tf.ensure_shape(temporal_weights_400, (None, None, 400))

        return temporal_weights_400  # Will be used for multiplication with news embeddings
 

class NRMSTemporalModel_Layer:
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)
    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference
    on Natural Language Processing (EMNLP-IJCNLP)
    Attributes:
    """
    def __init__(
        self,
        hparams: dict,
        word2vec_embedding: np.ndarray = None,
        word_emb_dim: int = 300,
        vocab_size: int = 32000,
        seed: int = None,
    ):
        """Initialization steps for NRMS."""
        self.hparams = hparams
        self.seed = seed
        # SET SEED:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        # INIT THE WORD-EMBEDDINGS:
        if word2vec_embedding is None:
            # Xavier Initialization
            initializer = GlorotUniform(seed=self.seed)
            self.word2vec_embedding = initializer(shape=(vocab_size, word_emb_dim))
            # self.word2vec_embedding = np.random.rand(vocab_size, word_emb_dim)
        else:
            self.word2vec_embedding = word2vec_embedding
        # BUILD AND COMPILE MODEL:
        self.model, self.scorer = self._build_graph()
        data_loss = self._get_loss(self.hparams.loss)
        train_optimizer = self._get_opt(
            optimizer=self.hparams.optimizer, lr=self.hparams.learning_rate
        )
        self.model.compile(loss=data_loss, optimizer=train_optimizer)
    def _get_loss(self, loss: str):
        """Make loss function, consists of data loss and regularization loss
        Returns:
            object: Loss function or loss function name
        """
        if loss == "cross_entropy_loss":
            data_loss = "categorical_crossentropy"
        elif loss == "log_loss":
            data_loss = "binary_crossentropy"
        else:
            raise ValueError(f"this loss not defined {loss}")
        return data_loss
    def _get_opt(self, optimizer: str, lr: float):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            object: An optimizer.
        """
        # TODO: shouldn't be a string input you should just set the optimizer, to avoid stuff like this:
        # => 'WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.'
        if optimizer == "adam":
            train_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            raise ValueError(f"this optimizer not defined {optimizer}")
        return train_opt
    def _build_graph(self):
        """Build NRMS model and scorer.
        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        model, scorer = self._build_nrms()
        return model, scorer
    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NRMS.
        Args:
            titleencoder (object): the news encoder of NRMS.
        Return:
            object: the user encoder of NRMS.
        """
        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="int32"
        )
        click_title_presents = tf.keras.layers.TimeDistributed(titleencoder)(
            his_input_title
        )
        y = SelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)(
            [click_title_presents] * 3
        )
        user_present = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)
        model = tf.keras.Model(his_input_title, user_present, name="user_encoder")
        return model
    def _build_newsencoder(self):
        """The main function to create news encoder of NRMS.
        Args:
            embedding_layer (object): a word embedding layer.
        Return:
            object: the news encoder of NRMS.
        """
        embedding_layer = tf.keras.layers.Embedding(
            self.word2vec_embedding.shape[0],
            self.word2vec_embedding.shape[1],
            weights=[self.word2vec_embedding],
            trainable=True,
        )
        sequences_input_title = tf.keras.Input(
            shape=(self.hparams.title_size,), dtype="int32"
        )
        embedded_sequences_title = embedding_layer(sequences_input_title)
        y = tf.keras.layers.Dropout(self.hparams.dropout)(embedded_sequences_title)
        y = SelfAttention(self.hparams.head_num, self.hparams.head_dim, seed=self.seed)(
            [y, y, y]
        )
        # Create configurable Dense layers:
        for layer in [400, 400, 400]:
            y = tf.keras.layers.Dense(units=layer, activation="relu")(y)
            y = tf.keras.layers.BatchNormalization()(y)
            y = tf.keras.layers.Dropout(self.hparams.dropout)(y)
        y = tf.keras.layers.Dropout(self.hparams.dropout)(y)
        pred_title = AttLayer2(self.hparams.attention_hidden_dim, seed=self.seed)(y)
        model = tf.keras.Model(sequences_input_title, pred_title, name="news_encoder")
        return model
    def _build_nrms(self):

        """Build NRMS model with learned temporal features.

        Instead of using pre-computed temporal discounts, this version learns

        temporal relationships from raw time differences.

        """

        # Input layers

        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size),
            dtype="int32",
        )

        pred_input_title = tf.keras.Input(
            shape=(None, self.hparams.title_size),
            dtype="int32",
        )

        pred_input_title_one = tf.keras.Input(
            shape=(1, self.hparams.title_size),
            dtype="int32",
        )

        # Time delta inputs (now just raw time differences)

        time_delta = tf.keras.Input(
            shape=(None, 1), dtype="float32"
        )

        time_delta_one = tf.keras.Input(
            shape=(1, 1), dtype="float32"
        )
    
        # Reshape single prediction input

        pred_title_one_reshape = tf.keras.layers.Reshape(
            (self.hparams.title_size,)
        )(pred_input_title_one)

        # Build encoders

        titleencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder(titleencoder)
        self.newsencoder = titleencoder
    
        # Get user representation

        user_present = self.userencoder(his_input_title)
        # Get news representations

        news_present = tf.keras.layers.TimeDistributed(self.newsencoder)(
            pred_input_title
        )

        news_present_one = self.newsencoder(pred_title_one_reshape)
        # Create temporal layer

        temporal_layer = TemporalLayer(units=64, name='temporal_layer')
        # Learn temporal weights and apply them
        temporal_weights = temporal_layer(time_delta)
        temporal_weights_one = temporal_layer(time_delta_one)
        # Apply temporal weights to news representations

        news_present = tf.keras.layers.Multiply()([news_present, temporal_weights])
        news_present_one = tf.keras.layers.Multiply()([news_present_one, temporal_weights_one])

        # Compute final predictions

        preds = tf.keras.layers.Dot(axes=-1)([news_present, user_present])
        preds = tf.keras.layers.Activation(activation="softmax")(preds)
        pred_one = tf.keras.layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = tf.keras.layers.Activation(activation="sigmoid")(pred_one)

        # Create models
        model = tf.keras.Model(
            [his_input_title, pred_input_title, time_delta],
            preds
        )

        scorer = tf.keras.Model(
            [his_input_title, pred_input_title_one, time_delta_one],
            pred_one
        )
    
        return model, scorer
 
    