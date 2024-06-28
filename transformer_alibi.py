# Code reference: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionHead(nn.Module):
    """
    One head of self-attention
    """

    def __init__(self, n_embd, head_size, block_size, m, dropout_prob):
        """
        :param n_embd: Embedding dimension
        :param head_size: Size of each attention head
        :param block_size: Maximum context length for predictions
        :param m: Numerical constant for bias matrix
        :param dropout_prob: Dropout probability
        """

        super().__init__()
        self.w_key = nn.Linear(in_features=n_embd, out_features=head_size, bias=False)
        self.w_query = nn.Linear(in_features=n_embd, out_features=head_size, bias=False)
        self.w_value = nn.Linear(in_features=n_embd, out_features=head_size, bias=False)
        self.m = m
        self.register_buffer(name="tril", tensor=torch.tril(input=torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, is_decoder=False):  # Compute single head attention
        """
        Forward pass

        :param x: Input tensor
        :param is_decoder: Is decoder?
        :return: Output tensor, attention weights
        """

        key, query, value = self.w_key(x), self.w_query(x), self.w_value(x)

        # Implement ALiBi
        right = torch.arange(x.shape[1])[None, :]
        left = torch.arange(x.shape[1])[:, None]
        bias = (self.m * (right - left)).unsqueeze(0)

        attention_head_weights = (query @ key.transpose(-2, -1) * (key.shape[-1] ** -0.5)) + bias

        if is_decoder:  # Mask future tokens
            T = x.shape[1]
            attention_head_weights = attention_head_weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        attention_head_weights = F.softmax(input=attention_head_weights, dim=-1)
        attention_head_weights = self.dropout(attention_head_weights)

        y = attention_head_weights @ value
        return y, attention_head_weights


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel
    """

    def __init__(self, n_embd, n_head, head_size, block_size, dropout_prob):
        """
        :param n_embd: Embedding dimension
        :param n_head: Number of attention heads
        :param head_size: Size of each attention head
        :param block_size: Maximum context length for predictions
        :param dropout_prob: Dropout probability
        """

        super().__init__()

        # Code reference:
        # https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L754
        def get_slopes(n):
            """
            Find the values of m given the number of heads

            :param n: Number of heads
            :return: Value of m for each head
            """

            def get_slopes_power_of_2(n):
                """
                Find the values of m given the number of heads as a power of 2

                :param n: Number of heads
                :return: Value of m for each head
                """

                start = (2 ** (-2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(
                    2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

        self.register_buffer("m", torch.Tensor(get_slopes(n_head)))
        self.attention_heads = nn.ModuleList([AttentionHead(
            n_embd=n_embd,
            head_size=head_size,
            block_size=block_size,
            m=self.m[i],
            dropout_prob=dropout_prob) for i in range(n_head)])
        self.w_out = nn.Linear(in_features=head_size * n_head, out_features=n_embd)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, is_decoder=False):  # Compute multi head attention
        """
        Forward pass

        :param x: Input tensor
        :param is_decoder: Is decoder?
        :return: Output tensor, attention weights
        """

        outputs = []
        multi_head_attention_weights = []
        for attention_head in self.attention_heads:
            out, multi_head_attention_weight = attention_head(x, is_decoder)
            outputs.append(out)
            multi_head_attention_weights.append(multi_head_attention_weight)

        out = torch.cat(tensors=outputs, dim=-1)
        out = self.dropout(self.w_out(out))
        return out, multi_head_attention_weights


class FeedForward(nn.Module):
    def __init__(self, n_embd, n_hidden, dropout_prob):
        """
        :param n_embd: Embedding dimension
        :param n_hidden: Hidden layer dimension
        :param dropout_prob: Dropout probability
        """

        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(in_features=n_embd, out_features=n_hidden),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_embd),
            nn.Dropout(p=dropout_prob))

    def forward(self, x):
        """
        Forward pass

        :param x: Input tensor
        :return: Output tensor
        """

        return self.ff(x)


class Block(nn.Module):
    """
    Single Block of a Transformer
    """

    def __init__(self, n_embd, n_head, block_size, n_hidden, dropout_prob):
        """
        :param n_embd: Embedding dimension
        :param n_head: Number of attention heads
        :param block_size: Maximum context length for predictions
        :param n_hidden: Hidden layer dimension
        :param dropout_prob: Dropout probability
        """

        super().__init__()
        head_size = n_embd // n_head
        self.block_attention = MultiHeadAttention(
            n_embd=n_embd,
            n_head=n_head,
            head_size=head_size,
            block_size=block_size,
            dropout_prob=dropout_prob)
        self.feed_forward = FeedForward(n_embd=n_embd, n_hidden=n_hidden, dropout_prob=dropout_prob)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=n_embd)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=n_embd)

    def forward(self, x, is_decoder=False):  # Use pre-layer norm
        """
        Forward pass

        :param x: Input tensor
        :param is_decoder: Is decoder?
        :return: Output tensor, attention weights
        """

        out, block_attention_weights = self.block_attention(self.layer_norm1(x), is_decoder)
        x = x + out  # residual connection
        x = x + self.feed_forward(self.layer_norm2(x))
        return x, block_attention_weights


class Encoder(nn.Module):
    """
    Transformer Encoder
    """

    def __init__(self, vocab_size, n_layer, n_embd, n_head, block_size, n_hidden, dropout_prob):
        """
        :param vocab_size: Vocab size
        :param n_layer: Number of transformer blocks
        :param n_embd: Embedding dimension
        :param n_head: Number of attention heads
        :param block_size: Maximum context length for predictions
        :param n_hidden: Hidden layer dimension
        :param dropout_prob: Dropout probability
        """

        super().__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
        self.blocks = nn.ModuleList([Block(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            n_hidden=n_hidden,
            dropout_prob=dropout_prob) for _ in range(n_layer)])

    def forward(self, idx):
        """
        Forward pass

        :param idx: Input tensor
        :return: Output tensor, attention weights
        """

        x = self.token_embedding_table(idx)

        block_attention_weights = []
        for block in self.blocks:
            x, self_attention_weight = block(x, is_decoder=False)
            block_attention_weights.extend(self_attention_weight)
        return x, block_attention_weights


class Decoder(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, vocab_size, n_layer, n_embd, n_heads, block_size, n_hidden,
                 dropout_prob, use_init_weights=False):
        """
        :param vocab_size: Vocab size
        :param n_layer: Number of transformer blocks
        :param n_embd: Embedding dimension
        :param n_heads: Number of attention heads
        :param block_size: Maximum context length for predictions
        :param n_hidden: Hidden layer dimension
        :param dropout_prob: Dropout probability
        :param use_init_weights: Use weights initialized from normal distribution
        """

        super().__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
        self.blocks = nn.ModuleList([Block(
            n_embd=n_embd,
            n_head=n_heads,
            block_size=block_size,
            n_hidden=n_hidden,
            dropout_prob=dropout_prob) for _ in range(n_layer)])
        self.layer_norm_final = nn.LayerNorm(normalized_shape=n_embd)
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size)

        if use_init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights by sampling from a normal distribution with a mean of 0 and a standard deviation of 0.05.
        :param module: Module to initialize
        :return:
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.05)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.05)

    def forward(self, idx, targets=None):
        """
        Forward pass

        :param idx: Input tensor
        :param targets: Labels
        :return: Output tensor, attention weights
        """
        x = self.token_embedding_table(idx)

        block_attention_weights = []
        for block in self.blocks:
            x, self_attention_weight = block(x, is_decoder=True)
            block_attention_weights.extend(self_attention_weight)

        x = self.layer_norm_final(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            num_batches, num_tokens, num_channels = logits.shape
            logits = logits.view(num_batches * num_tokens, num_channels)
            targets = targets.view(num_batches * num_tokens)
            loss = F.cross_entropy(logits, targets)

        return loss, block_attention_weights


class Classifier(nn.Module):
    def __init__(self, vocab_size, n_layer, n_embd, n_head, block_size, dropout_prob,
                 n_input, n_hidden, n_output, use_init_weights=False):
        """
        :param vocab_size: Vocab size
        :param n_layer: Number of transformer blocks
        :param n_embd: Embedding dimension
        :param n_head: Number of attention heads
        :param block_size: Maximum context length for predictions
        :param dropout_prob: Dropout probability
        :param n_input: Input size for the classifier, should match the embedding size of the transformer
        :param n_hidden: Hidden size for the classifier
        :param n_output: Output size for the classifier, we have 3 classes
        :param use_init_weights: Use weights initialized from normal distribution
        """

        super().__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            n_hidden=n_hidden,
            dropout_prob=dropout_prob)
        self.fc1 = nn.Linear(in_features=n_input, out_features=n_hidden)
        self.fc2 = nn.Linear(in_features=n_hidden, out_features=n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

        if use_init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights by sampling from a normal distribution with a mean of 0 and a standard deviation of 0.05.
        :param module: Module to initialize
        :return:
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.05)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.05)

    def forward(self, x):
        """
        Forward pass

        :param x: Input tensor
        :return: Output tensor, attention weights
        """

        x, block_attention_weights = self.encoder(x)
        x = torch.mean(x, dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x, block_attention_weights
