import torch
from torch import nn


class NMT_Transformer(nn.Module):
    def __init__(self, vocab_size:int, dim_embed:int,
                 dim_model:int, dim_feedforward:int, num_layers:int,
                 dropout_probability:float, maxlen:int):
        """
        Neural Machine Translation Transformer model.
        
        Args:
            vocab_size: Size of the vocabulary
            dim_embed: Dimension of the embedding vectors
            dim_model: Dimension of the model (hidden size)
            dim_feedforward: Dimension of the feedforward network in transformer layers
            num_layers: Number of encoder and decoder layers
            dropout_probability: Dropout rate
            maxlen: Maximum sequence length for positional embeddings
        """
        super().__init__()

        # Shared embeddings for source and target tokens
        self.embed_shared_src_trg_cls = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim_embed)
        # Shared positional embeddings for source and target sequences
        self.positonal_shared_src_trg = nn.Embedding(num_embeddings=maxlen, embedding_dim=dim_embed)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_probability)

        # Create encoder layer with specified parameters
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=8,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout_probability,
                                                   batch_first=True, norm_first=True)
        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # Create decoder layer with specified parameters
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=8,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout_probability,
                                                   batch_first=True, norm_first=True)
        # Stack multiple decoder layers
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection layer to vocabulary size
        self.classifier = nn.Linear(dim_model, vocab_size)
        # Weight sharing between embedding and output projection (tied embeddings)
        self.classifier.weight = self.embed_shared_src_trg_cls.weight

        self.maxlen = maxlen
        # Initialize model weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights of the model using standard transformer initialization.
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, source, target, pad_tokenId):
        """
        Forward pass of the NMT Transformer.
        
        Args:
            source: Source sequence tensor [batch_size, source_seq_len]
            target: Target sequence tensor [batch_size, target_seq_len]
            pad_tokenId: Token ID used for padding
            
        Returns:
            logits: Output logits [batch_size, target_seq_len, vocab_size]
            loss: Cross-entropy loss if target length > 1, otherwise None
        """
        # Get batch size and sequence lengths
        B, Ts = source.shape
        B, Tt = target.shape
        device = source.device
        
        ## Encoder Path
        # Create positional embeddings for source sequence
        src_poses = self.positonal_shared_src_trg(torch.arange(0, Ts).to(device).unsqueeze(0).repeat(B, 1))
        # Combine token embeddings with positional embeddings and apply dropout
        src_embedings = self.dropout(self.embed_shared_src_trg_cls(source) + src_poses)

        # Create padding mask for source sequence
        src_pad_mask = source == pad_tokenId
        # Pass through encoder to get memory
        memory = self.transformer_encoder(src=src_embedings, mask=None, src_key_padding_mask=src_pad_mask, is_causal=False)
        
        ## Decoder Path
        # Create positional embeddings for target sequence
        trg_poses = self.positonal_shared_src_trg(torch.arange(0, Tt).to(device).unsqueeze(0).repeat(B, 1))
        # Combine token embeddings with positional embeddings and apply dropout
        trg_embedings = self.dropout(self.embed_shared_src_trg_cls(target) + trg_poses)
        
        # Create padding mask for target sequence
        trg_pad_mask = target == pad_tokenId
        # Create causal mask to prevent attending to future tokens
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(Tt, dtype=bool).to(device)
        # Pass through decoder
        decoder_out = self.transformer_decoder.forward(tgt=trg_embedings,
                                                memory=memory,
                                                tgt_mask=tgt_mask,
                                                memory_mask=None,
                                                tgt_key_padding_mask=trg_pad_mask,
                                                memory_key_padding_mask=None)
        
        ## Classifier Path
        # Project decoder output to vocabulary space
        logits = self.classifier(decoder_out)
        
        # Calculate loss if we have more than one target token
        loss = None
        if Tt > 1:
            # For model logits we need all tokens except the last one
            flat_logits = logits[:,:-1,:].reshape(-1, logits.size(-1))
            # For targets we need all tokens except the first one (shift right)
            flat_targets = target[:,1:].reshape(-1)
            # Calculate cross-entropy loss, ignoring padding tokens
            loss = nn.functional.cross_entropy(flat_logits, flat_targets, ignore_index=pad_tokenId)
        return logits, loss