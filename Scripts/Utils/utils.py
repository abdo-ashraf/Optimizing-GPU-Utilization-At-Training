import torch
from Utils.NMT_Transformer import NMT_Transformer

def setup_data(batch_size=256, num_batch=50):
    """
    Set up random data for training.
    
    Args:
        batch_size: Size of each batch
        num_batch: Number of batches to generate
        
    Returns:
        data_x: Source data
        data_y: Target data
    """
    data_x = torch.randint(0, 5000, (num_batch, batch_size, 50))
    data_y = torch.randint(0, 5000, (num_batch, batch_size, 20))
    return data_x, data_y

def setup_model():
    """
    Set up model with default parameters.
    
    Returns:
        Model with default parameters
    """
    vocab_size = 5000
    dim_embed = 512
    dim_model = 512
    dim_feedforward = 512*4
    num_layers = 3
    dropout_probability = 0.1
    maxlen = 50
    
    model = NMT_Transformer(
        vocab_size=vocab_size,
        dim_embed=dim_embed,
        dim_model=dim_model,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        dropout_probability=dropout_probability,
        maxlen=maxlen
    )
    
    return model 