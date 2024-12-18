import torch
import torch.nn as nn

class SoftAttention(nn.Module):
    """
    A soft attention mechanism that computes a weighted sum of input features
    based on learned attention weights.

    Attributes:
        W (nn.Linear): Linear layer for projecting the input to the attention dimension.
        q (nn.Parameter): Query vector used for calculating attention scores.
        softmax (nn.Softmax): Softmax function for normalizing attention scores.
    """
    def __init__(self, input_dim, attention_dim, seed=0):
        """
        Initializes the SoftAttention module.

        Args:
            input_dim (int): Dimension of the input features.
            attention_dim (int): Dimension of the attention space.
            seed (int): Random seed for initialization (default: 0).
        """
        super(SoftAttention, self).__init__()
        self.W = nn.Linear(input_dim, attention_dim, bias=True)
        self.q = nn.Parameter(torch.randn(attention_dim))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, mask=None):
        """
        Forward pass of the SoftAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            mask (torch.Tensor, optional): Optional mask tensor (not used in this implementation).

        Returns:
            torch.Tensor: Weighted sum of input features, shape (batch_size, input_dim).
        """
        # Compute attention scores using a linear projection and a query vector.
        attention = torch.tanh(self.W(x))  # (batch_size, seq_len, attention_dim)
        attention = torch.matmul(attention, self.q)  # (batch_size, seq_len)

        # Compute attention weights using softmax.
        attention_weights = self.softmax(attention)  # (batch_size, seq_len)

        # Reshape attention weights for batch matrix multiplication.
        attention_weights = attention_weights.unsqueeze(1)  # (batch_size, 1, seq_len)

        # Compute weighted sum of input features.
        weighted_sum = torch.bmm(attention_weights, x)  # (batch_size, 1, input_dim)

        # Remove the extra dimension.
        weighted_sum = weighted_sum.squeeze(1)  # (batch_size, input_dim)

        return weighted_sum

class MultiheadAttention(nn.Module):
    """
    A multi-head attention mechanism that computes context-aware representations
    using multiple attention heads.

    Attributes:
        multiheads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        output_dim (int): Total output dimension (multiheads * head_dim).
        W_query (nn.Parameter): Weight matrix for query transformation.
        W_key (nn.Parameter): Weight matrix for key transformation.
        W_value (nn.Parameter): Weight matrix for value transformation.
        softmax (nn.Softmax): Softmax function for attention score normalization.
    """
    def __init__(self, multiheads, head_dim, input_dim, seed=0):
        """
        Initializes the MultiheadAttention module.

        Args:
            multiheads (int): Number of attention heads.
            head_dim (int): Dimension of each attention head.
            input_dim (int): Dimension of the input features.
            seed (int): Random seed for initialization (default: 0).
        """
        super(MultiheadAttention, self).__init__()
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim

        # Initialize learnable weight matrices for queries, keys, and values.
        self.W_query = nn.Parameter(torch.empty(input_dim, self.output_dim))
        self.W_key = nn.Parameter(torch.empty(input_dim, self.output_dim))
        self.W_value = nn.Parameter(torch.empty(input_dim, self.output_dim))

        # Xavier initialization for weight matrices.
        nn.init.xavier_uniform_(self.W_query)
        nn.init.xavier_uniform_(self.W_key)
        nn.init.xavier_uniform_(self.W_value)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V):
        """
        Forward pass of the MultiheadAttention module.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, seq_len, input_dim).
            K (torch.Tensor): Key tensor of shape (batch_size, seq_len, input_dim).
            V (torch.Tensor): Value tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor after multi-head attention, shape (batch_size, seq_len, output_dim).
        """
        # Transform queries, keys, and values using weight matrices.
        Q_seq = torch.matmul(Q, self.W_query)
        K_seq = torch.matmul(K, self.W_key)
        V_seq = torch.matmul(V, self.W_value)

        # Reshape for multi-head attention: (batch_size, seq_len, multiheads, head_dim).
        Q_seq = Q_seq.view(-1, Q_seq.shape[1], self.multiheads, self.head_dim).permute(0, 2, 1, 3)
        K_seq = K_seq.view(-1, K_seq.shape[1], self.multiheads, self.head_dim).permute(0, 2, 1, 3)
        V_seq = V_seq.view(-1, V_seq.shape[1], self.multiheads, self.head_dim).permute(0, 2, 1, 3)

        # Compute scaled dot-product attention scores.
        A = torch.matmul(Q_seq, K_seq.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=Q_seq.device))
        A = self.softmax(A)  # Normalize scores.

        # Compute attention-weighted values.
        O_seq = torch.matmul(A, V_seq)  # (batch_size, multiheads, seq_len, head_dim)
        O_seq = O_seq.permute(0, 2, 1, 3)  # Rearrange dimensions.

        # Concatenate attention heads into a single output tensor.
        O_seq = O_seq.reshape(-1, O_seq.size(1), self.output_dim)  # (batch_size, seq_len, output_dim)

        return O_seq
