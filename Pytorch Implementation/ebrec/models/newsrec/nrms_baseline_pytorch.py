import torch.nn as nn
import torch 
import numpy as np
from ebrec.models.newsrec.layers_pytorch import SoftAttention, MultiheadAttention

class NewsEncoder(nn.Module):
    def __init__(self, word2vec_embedding, title_size, dropout_rate, num_attention_heads, head_dim, word_embedding_dim, attention_hidden_dim, seed):
        super(NewsEncoder, self).__init__()
        # Create embedding layer using pre-trained word2vec embeddings
        self.embedding_layer = nn.Embedding.from_pretrained(word2vec_embedding, freeze=False)

        # Add Dropout layer 1 with the given dropout rate
        self.dropout_layer1 = nn.Dropout(p=dropout_rate)

        # Add Dropout layer 2 with the given dropout rate
        self.dropout_layer2 = nn.Dropout(p=dropout_rate)

        # Set title size as the input length
        self.title_size = title_size

        # Initialize multihead attention
        self.multihead_attn = MultiheadAttention(num_attention_heads, head_dim, word_embedding_dim, seed)

        # Initalize soft alignment attention
        self.soft_attention_layer = SoftAttention(num_attention_heads * head_dim, attention_hidden_dim, seed)


    def forward(self, sequences_input_title):
        sequences_input_title = sequences_input_title.long().cuda()
        embedded_sequences_title = self.embedding_layer(sequences_input_title) # Run input sequence through embedding layer

        embedded_sequences_title_with_dropout = self.dropout_layer1(embedded_sequences_title) # Apply dropout layer

        attn_output = self.multihead_attn(embedded_sequences_title_with_dropout, embedded_sequences_title_with_dropout , embedded_sequences_title_with_dropout) # Apply multihead attention

        att_output_with_dropout = self.dropout_layer2(attn_output)

        output = self.soft_attention_layer(att_output_with_dropout) # Apply soft alignment attention

        return output

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
    
class NRMS_baseline_pytorch(nn.Module):
    """
    A PyTorch implementation of the Neural News Recommendation System (NRMS) baseline model.

    This model is designed for news recommendation tasks and uses a news encoder and a user encoder 
    with attention mechanisms to encode user history and candidate news. The model outputs a 
    prediction (probability) for each candidate news article based on the user's history.

    Args:
        hparams (dict): Hyperparameters including the size of news titles, dropout rate, number of attention heads, 
                         attention head dimension, word embedding dimension, and attention hidden dimension.
        word2vec_embedding (np.ndarray, optional): Pre-trained word2vec embeddings for the news encoder. Default is None.
        seed (int, optional): Random seed for reproducibility. Default is 1.
        word_emb_dim (int, optional): Dimension of the word embeddings. Default is 768.
        vocab_size (int, optional): Size of the vocabulary. Default is 250002.

    Example:
        # Initialize hyperparameters
        hparams = {
            'title_size': 50,
            'dropout_rate': 0.2,
            'num_attention_heads': 4,
            'head_dim': 64,
            'word_embedding_dim': 768,
            'attention_hidden_dim': 512
        }

        # Initialize model
        model = NRMS_baseline_pytorch(hparams=hparams)

        # Forward pass with example input
        his_input_title = torch.randint(0, 100, (32, 20))  # Example user history input (batch_size, history_size)
        pred_input_title = torch.randint(0, 100, (32, 10, 20))  # Example candidate news input (batch_size, num_candidates, title_size)
        preds = model(his_input_title, pred_input_title)
    """
    
    def __init__(self, hparams: dict, word2vec_embedding: np.ndarray = None, seed: int = 1, word_emb_dim: int = 768, vocab_size: int = 250002):
        """
        Initializes the NRMS model components and sets the random seed for reproducibility.

        Args:
            hparams (dict): Hyperparameters for the model.
            word2vec_embedding (np.ndarray, optional): Pre-trained word2vec embeddings for the news encoder.
            seed (int, optional): Random seed for reproducibility.
            word_emb_dim (int, optional): Dimension of word embeddings.
            vocab_size (int, optional): Size of the vocabulary.
        """
        super(NRMS_baseline_pytorch, self).__init__()

        # Store the random seed for reproducibility
        self.seed = seed
        
        # Initialize the news encoder and user encoder using the given hyperparameters
        self.news_encoder = self._build_news_encoder(word2vec_embedding, hparams.title_size, hparams.dropout,
                                                     hparams.head_num, hparams.head_dim,
                                                     hparams.word_embedding_dim, hparams.attention_hidden_dim, seed)
        self.user_encoder = self._build_user_encoder(self.news_encoder, hparams.head_num, 
                                                     hparams.head_dim, hparams.attention_hidden_dim, seed)
        self.softmax = nn.Softmax(dim=-1)

        # Set random seeds for PyTorch and NumPy
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize word2vec embeddings if not provided
        if word2vec_embedding is None:
            self.word2vec_embedding = np.random.rand(vocab_size, word_emb_dim)
        else:
            self.word2vec_embedding = word2vec_embedding

    def _build_news_encoder(self, word2vec_embedding, title_size, dropout_rate, num_attention_heads, head_dim, word_embedding_dim, attention_hidden_dim, seed):
        """
        Builds the news encoder using the specified hyperparameters.

        Args:
            word2vec_embedding (np.ndarray): Pre-trained word embeddings.
            title_size (int): The size of the news title.
            dropout_rate (float): Dropout rate to be used in the model.
            num_attention_heads (int): The number of attention heads.
            head_dim (int): The dimension of each attention head.
            word_embedding_dim (int): The dimension of the word embeddings.
            attention_hidden_dim (int): The hidden dimension for attention layers.
            seed (int): Random seed.

        Returns:
            NewsEncoder: The news encoder object.
        """
        return NewsEncoder(word2vec_embedding, title_size, dropout_rate, num_attention_heads, head_dim,
                           word_embedding_dim, attention_hidden_dim, seed)

    def _build_user_encoder(self, news_encoder, num_attention_heads, head_dim, attention_hidden_dim, seed):
        """
        Builds the user encoder using the specified hyperparameters and a pre-trained news encoder.

        Args:
            news_encoder (NewsEncoder): The news encoder object.
            num_attention_heads (int): The number of attention heads.
            head_dim (int): The dimension of each attention head.
            attention_hidden_dim (int): The hidden dimension for attention layers.
            seed (int): Random seed.

        Returns:
            UserEncoder: The user encoder object.
        """
        return UserEncoder(news_encoder, num_attention_heads, head_dim, attention_hidden_dim, seed)

    def forward(self, his_input_title, pred_input_title, validation=False):
        """
        Forward pass for the NRMS model. Computes the similarity between the user's history and the candidate news.

        Args:
            his_input_title (Tensor): A tensor containing the user's historical news titles, 
                                       shape (batch_size, history_size).
            pred_input_title (Tensor): A tensor containing the candidate news titles, 
                                        shape (batch_size, num_candidates, title_size).
            validation (bool, optional): Whether the forward pass is for validation. Default is False.

        Returns:
            Tensor: A tensor containing the predicted probabilities for each candidate news, 
                    shape (batch_size, num_candidates).
        """
        # Encode user history (shape: (batch_size, embedding_dim))
        user_present = self.user_encoder(his_input_title)

        # Flatten the candidate news titles for batch processing (shape: (batch_size * num_candidates, title_size))
        batch_size, num_candidates, title_size = pred_input_title.size()
        pred_input_flat = pred_input_title.view(-1, title_size)

        # Encode candidate news (shape: (batch_size * num_candidates, embedding_dim))
        news_present_flat = self.news_encoder(pred_input_flat)

        # Reshape back to (batch_size, num_candidates, embedding_dim)
        embedding_dim = news_present_flat.size(1)
        news_present = news_present_flat.view(batch_size, num_candidates, embedding_dim)

        # Compute dot product between user and news embeddings (shape: (batch_size, num_candidates))
        preds = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)

        # Apply softmax to get probabilities for each candidate (shape: (batch_size, num_candidates))
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
        pred_title_one_reshape = pred_input_title_one.reshape(-1, 30)

        # Encode user history
        user_present = self.user_encoder(his_input_title)

        # Encode a single candidate news title
        news_present_one = self.news_encoder(pred_title_one_reshape)

        # Compute relevance score
        preds_one = torch.sum(user_present * news_present_one, dim=1, keepdim=True)

        # Apply sigmoid activation
        preds_one = self.activation(preds_one)

        return preds_one