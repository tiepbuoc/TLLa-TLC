import torch
from utils.path import resource_path

# C\u1ed1 \u0111\u1ecbnh
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOTAL_WORDS = 17788+1
MAX_SEQUENCE_LEN = 32

# \u0110\u01b0\u1eddng d\u1eabn \u0111iểm kiểm tra m\u00f4 h\u00ecnh
BASE_MODEL_CHECKPOINT_PATH = resource_path("checkpoints/tllagpt-1.3.pth")

# C\u1ea5u h\u00ecnh m\u00f4 h\u00ecnh GPT
MODEL_SIZES = {
    'base': {
        'block_size': MAX_SEQUENCE_LEN,
        'vocab_size': TOTAL_WORDS,
        'n_layer': 8,
        'n_head': 8,
        'n_embd': 256,
    }
}
