import torch.nn as nn
import torch 
import numpy as np
from ebrec.models.newsrec.layers_pytorch import SoftAttention, MultiheadAttention

class NewsEncoder(nn.Module):
    """
    Encodes news titles into fixed-length document vectors using a sequence of dense layers.

    Attributes:
        hparams: Hyperparameters object containing configurations like title size, head dimensions, and dropout rate.
        units_per_layer: List specifying the number of units in each dense layer.
        document_vector_dim: Dimension of the input document vectors.
        output_dim: Dimension of the final encoded representation.
        dense_layers: Sequential dense layers with ReLU, BatchNorm, and Dropout.
        output_layer: Final dense layer to produce the output representation.
        l2_regularization: Weight decay for L2 regularization.
    """
    def __init__(self, hparams, units_per_layer=[512, 512, 512]):
        super(NewsEncoder, self).__init__()
        self.hparams = hparams
        self.units_per_layer = units_per_layer
        self.document_vector_dim = self.hparams.title_size
        self.output_dim = self.hparams.head_num * self.hparams.head_dim

        # Initialize dense layers
        self.dense_layers = nn.ModuleList()
        in_units = self.document_vector_dim

        for units in self.units_per_layer:
            self.dense_layers.append(
                nn.Sequential(
                    nn.Linear(in_units, units),
                    nn.ReLU(),
                    nn.BatchNorm1d(units),
                    nn.Dropout(self.hparams.dropout)
                )
            )
            in_units = units

        # Output layer
        self.output_layer = nn.Linear(in_units, self.output_dim)

        # L2 regularization weight decay
        self.l2_regularization = self.hparams.newsencoder_l2_regularization

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch_size, document_vector_dim).

        Returns:
            Encoded representation of shape (batch_size, output_dim).
        """
        # Pass through dense layers
        for layer in self.dense_layers:
            x = layer(x)

        # Final output layer
        x = self.output_layer(x)
        return x

class UserEncoder(nn.Module):
    """
    Encodes user behavior history into a fixed-length vector using attention mechanisms.

    Attributes:
        news_encoder: NewsEncoder instance to encode news titles.
        multihead_attn: MultiheadAttention layer for self-attention over user history.
        soft_attention_layer: SoftAttention layer for aggregating attended vectors.
    """
    def __init__(self, news_encoder, num_attention_heads, head_dim, attention_hidden_dim, seed):
        super(UserEncoder, self).__init__()

        # Initialize the news encoder
        self.news_encoder = news_encoder

        # Self Attention Layer
        self.multihead_attn = MultiheadAttention(num_attention_heads, head_dim, num_attention_heads * head_dim, seed)

        # Attention Layer
        self.soft_attention_layer = SoftAttention(input_dim=num_attention_heads * head_dim, 
                                                   attention_dim=attention_hidden_dim, 
                                                   seed=seed)

    def forward(self, his_input_title):
        """
        Forward pass to encode user history.

        Args:
            his_input_title: Tensor of shape (batch_size, history_size, title_size).

        Returns:
            Encoded user representation of shape (batch_size, embedding_dim).
        """
        # Flatten input for batch processing
        batch_size, history_size, title_size = his_input_title.size()
        his_input_title_flat = his_input_title.view(-1, title_size)  # Shape: (batch_size * history_size, title_size)

        # Encode all titles in user history
        news_present_flat = self.news_encoder(his_input_title_flat)  # Shape: (batch_size * history_size, embedding_dim)

        # Reshape back to (batch_size, history_size, embedding_dim)
        embedding_dim = news_present_flat.size(1)
        click_title_presents = news_present_flat.view(batch_size, history_size, embedding_dim)

        # Apply multihead self-attention
        y = self.multihead_attn(click_title_presents, click_title_presents, click_title_presents)

        # Aggregate representations using soft attention
        user_present = self.soft_attention_layer(y)

        return user_present
    
class NRMS_docvec_pytorch(nn.Module):
    """
    Neural Recommendation Model for News (NRMS) with document vectors.

    Attributes:
        hparams: Hyperparameters object for configuring the model.
        seed: Random seed for reproducibility.
        news_encoder: Encodes news titles into fixed-length vectors.
        user_encoder: Encodes user behavior history into a fixed-length vector.
        softmax: Applies softmax to the final prediction scores.
    """
    def __init__(self, hparams: dict, seed: int = 1):
        super(NRMS_docvec_pytorch, self).__init__()

        # Initialize seed
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize encoders
        self.news_encoder = self._build_news_encoder(hparams)
        self.user_encoder = self._build_user_encoder(self.news_encoder, 
                                                      hparams.head_num, 
                                                      hparams.head_dim, 
                                                      hparams.attention_hidden_dim, 
                                                      seed)
        self.softmax = nn.Softmax(dim=-1)

    def _build_news_encoder(self, hparams):
        """Helper method to create the NewsEncoder."""
        return NewsEncoder(hparams)

    def _build_user_encoder(self, news_encoder, num_attention_heads, head_dim, attention_hidden_dim, seed):
        """Helper method to create the UserEncoder."""
        return UserEncoder(news_encoder, num_attention_heads, head_dim, attention_hidden_dim, seed)

    def forward(self, his_input_title, pred_input_title, validation=False):
        """
        Forward pass for the NRMS model.

        Args:
            his_input_title: Tensor of shape (batch_size, history_size, title_size).
            pred_input_title: Tensor of shape (batch_size, num_candidates, title_size).
            validation: Boolean, indicates if the model is in validation mode.

        Returns:
            Predicted scores for candidate news of shape (batch_size, num_candidates).
        """
        # Encode user history
        user_present = self.user_encoder(his_input_title)

        # Flatten candidate news titles for batch processing
        batch_size, num_candidates, title_size = pred_input_title.size()
        pred_input_flat = pred_input_title.view(-1, title_size)  # Shape: (batch_size * num_candidates, title_size)

        # Encode candidate news
        news_present_flat = self.news_encoder(pred_input_flat)  # Shape: (batch_size * num_candidates, embedding_dim)

        # Reshape back to (batch_size, num_candidates, embedding_dim)
        embedding_dim = news_present_flat.size(1)
        news_present = news_present_flat.view(batch_size, num_candidates, embedding_dim)

        # Compute dot product for predictions
        preds = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)

        # Apply softmax to get probabilities
        preds = self.softmax(preds)
        return preds

class ScorerModel(nn.Module):
    """
    Scorer model for calculating relevance scores between user and news representations.

    Attributes:
        user_encoder: UserEncoder instance.
        news_encoder: NewsEncoder instance.
        activation: Sigmoid activation function for the output.
    """
    def __init__(self, user_encoder, news_encoder):
        super(ScorerModel, self).__init__()
        self.user_encoder = user_encoder
        self.news_encoder = news_encoder
        self.activation = nn.Sigmoid()

    def forward(self, his_input_title, pred_input_title_one):
        """
        Forward pass to calculate relevance scores.

        Args:
            his_input_title: Tensor of shape (batch_size, history_size, title_size).
            pred_input_title_one: Tensor of shape (batch_size, title_size).

        Returns:
            Relevance scores of shape (batch_size, 1).
        """
        # Reshape candidate news input
        pred_title_one_reshape = pred_input_title_one.reshape(-1, 768)

        # Encode user history
        user_present = self.user_encoder(his_input_title)

        # Encode a single candidate news title
        news_present_one = self.news_encoder(pred_title_one_reshape)

        # Compute relevance score
        preds_one = torch.sum(user_present * news_present_one, dim=1, keepdim=True)

        # Apply sigmoid activation
        preds_one = self.activation(preds_one)

        return preds_one