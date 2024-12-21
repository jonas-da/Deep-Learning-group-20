# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from ebrec.models.newsrec.layers import AttLayer2, SelfAttention
import tensorflow as tf
import numpy as np


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
    
class NRMSDocVec_temp_layer:
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
        seed: int = None,
        newsencoder_units_per_layer: list[int] = [512, 512, 512],

    ):
        """Initialization steps for NRMS."""
        self.hparams = hparams
        self.seed = seed
        self.newsencoder_units_per_layer = newsencoder_units_per_layer
        
        # SET SEED:
        tf.random.set_seed(seed)
        np.random.seed(seed)
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
            shape=(self.hparams.history_size, self.hparams.title_size), dtype="float32"
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

    def _build_newsencoder(self, units_per_layer: list[int] = list[512, 512, 512]):
        """THIS IS OUR IMPLEMENTATION.
        The main function to create a news encoder.

        Parameters:
            units_per_layer (int): The number of neurons in each Dense layer.

        Return:
            object: the news encoder.
        """
        DOCUMENT_VECTOR_DIM = self.hparams.title_size
        OUTPUT_DIM = self.hparams.head_num * self.hparams.head_dim

        # DENSE LAYERS (FINE-TUNED):
        sequences_input_title = tf.keras.Input(
            shape=(DOCUMENT_VECTOR_DIM,), dtype="float32"
        )
        x = sequences_input_title
        # Create configurable Dense layers:
        for layer in units_per_layer:
            x = tf.keras.layers.Dense(units=layer, activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.hparams.dropout)(x)

        # OUTPUT:
        pred_title = tf.keras.layers.Dense(units=OUTPUT_DIM, activation="relu")(x)

        # Construct the final model
        model = tf.keras.Model(
            inputs=sequences_input_title, outputs=pred_title, name="news_encoder"
        )

        return model

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """

        his_input_title = tf.keras.Input(
            shape=(self.hparams.history_size, self.hparams.title_size),
            dtype="float32",
        )
        pred_input_title = tf.keras.Input(
            # shape = (hparams.npratio + 1, hparams.title_size)
            shape=(None, self.hparams.title_size),
            dtype="float32",
        )
        pred_input_title_one = tf.keras.Input(
            shape=(
                1,
                self.hparams.title_size,
            ),
            dtype="float32",
        )
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

        titleencoder = self._build_newsencoder(
            units_per_layer=self.newsencoder_units_per_layer
        )

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
