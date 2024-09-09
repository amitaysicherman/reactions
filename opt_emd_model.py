from transformers import BertForTokenClassification, BertConfig
from preprocessing.to_edit_ops import OP_TO_ID
n_labels = len(OP_TO_ID)

def build_model_by_size_type(size, vocab_size, **kwargs):
    if size == "xs":
        hidden_size = 32
        num_hidden_layers = 2
        num_attention_heads = 2
        intermediate_size = 64
    elif size == "s":
        hidden_size = 64
        num_hidden_layers = 4
        num_attention_heads = 4
        intermediate_size = 128
    elif size == "m":
        hidden_size = 512
        num_hidden_layers = 8
        num_attention_heads = 8
        intermediate_size = 1024
    elif size == "l":
        hidden_size = 768
        num_hidden_layers = 12
        num_attention_heads = 12
        intermediate_size = 3072
    elif size == "xl":
        hidden_size = 1024
        num_hidden_layers = 12
        num_attention_heads = 16
        intermediate_size = 4096
    else:
        raise ValueError(f"Invalid size: {size}")

    return BertForTokenClassification(BertConfig(
        vocab_size=vocab_size,
        num_labels=n_labels,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        **kwargs
    ))


if __name__ == '__main__':
    for size in ["xs", "s", "m", "l", "xl"]:
        model = build_model_by_size_type(size, 200)
        trainables = sum(p.numel() for p in model.to("cpu").parameters() if p.requires_grad)
        print(f"Size: {size}, Trainable parameters: {trainables:,}")
