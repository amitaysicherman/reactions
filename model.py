from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
import numpy as np
from transformers.generation.utils import GenerateOutput

NO_META = 0
BOOLEAN_META = 1
LOOKUP_META = 2
ADD_META = 3


def get_layers(dims, dropout=0.0):
    layers = torch.nn.Sequential()
    for i in range(len(dims) - 1):
        layers.add_module(f"linear_{i}", torch.nn.Linear(dims[i], dims[i + 1]))
        layers.add_module(f"bn_{i}", torch.nn.BatchNorm1d(dims[i + 1]))
        if i < len(dims) - 2:
            layers.add_module(f"relu_{i}", torch.nn.ReLU())
        if dropout > 0:
            layers.add_module(f"dropout_{i}", torch.nn.Dropout(dropout))
    return layers


class CustomTranslationConfig(T5Config):
    def __init__(self, meta_type=NO_META, lookup_file="data/prot_emb.npy", lookup_len=4, **kwargs):
        super(CustomTranslationConfig, self).__init__(**kwargs)
        self.meta_type = meta_type
        self.lookup_file = lookup_file
        self.lookup_len = lookup_len


class CustomTranslationModel(T5ForConditionalGeneration):
    def __init__(self, config):
        super(CustomTranslationModel, self).__init__(config)
        meta_type = config.meta_type
        if meta_type != NO_META:
            self.meta_embedding = nn.Embedding(2, config.d_model)  # Embedding for meta (2 possible values)
        if meta_type == LOOKUP_META or meta_type == ADD_META:
            lookup_table = np.load(config.lookup_file)
            lookup_dim = lookup_table.shape[1]
            self.lookup_table = nn.Embedding.from_pretrained(torch.tensor(lookup_table), freeze=True).float()
            layers_dims = [lookup_dim] + [config.d_model] * config.lookup_len
            self.lookup_proj = get_layers(layers_dims, dropout=0.1)
        self.meta_type = meta_type

    def forward(self, input_ids=None, attention_mask=None, meta=None, labels=None, encoder_outputs=None, **kwargs):
        if self.meta_type == NO_META:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                encoder_outputs=encoder_outputs,
                **kwargs
            )

        if encoder_outputs is None:  # not none in generation
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            encoder_embedding = encoder_outputs.last_hidden_state
            emb_to_add = []

            if self.meta_type == BOOLEAN_META or self.meta_type == LOOKUP_META:
                meta_type = torch.where(meta == 0, torch.zeros_like(meta), torch.ones_like(meta))
                meta_type = meta_type.to(encoder_embedding.device)
                emb_to_add.append(self.meta_embedding(meta_type))
            if self.meta_type == LOOKUP_META or self.meta_type == ADD_META:
                meta_vector = self.lookup_table(meta)
                meta_vector = self.lookup_proj(meta_vector.squeeze(1)).unsqueeze(1)
                emb_to_add.append(meta_vector)

            if self.meta_type == BOOLEAN_META or self.meta_type == LOOKUP_META:
                combined_embedding = torch.cat([encoder_embedding] + emb_to_add, dim=1)
                ones_size = len(emb_to_add)
                ones_for_mask = torch.ones((attention_mask.shape[0], ones_size), device=combined_embedding.device)
                attention_mask = torch.cat([attention_mask, ones_for_mask], dim=-1)
                encoder_outputs.last_hidden_state = combined_embedding
            else:  # self.meta_type==ADD_META
                assert len(emb_to_add) == 1
                encoder_outputs.last_hidden_state = encoder_embedding + emb_to_add[0]
        outputs = super().forward(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask.float(),
            labels=labels,
            **kwargs
        )
        return outputs

    def generate(
            self,
            meta: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        return super().generate(meta=meta, **kwargs)

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            decoder_attention_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        meta = kwargs.pop("meta", None)
        results = super().prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, head_mask,
                                                        decoder_head_mask, decoder_attention_mask, cross_attn_head_mask,
                                                        use_cache, encoder_outputs, **kwargs)
        results["meta"] = meta
        return results


def build_model_by_size_type(size, meta_type=NO_META, **kwargs):
    if size == "xs":
        d_model = 32
        ff = 64
        n_heads = 2
        n_layers = 2
        lookup_len = 1

    elif size == "s":
        d_model = 64
        ff = 128
        n_heads = 4
        n_layers = 4
        lookup_len = 2


    elif size == "m":
        d_model = 256
        ff = 512
        n_heads = 8
        n_layers = 6
        lookup_len = 4



    elif size == "l":
        d_model = 512
        ff = 1024
        n_heads = 8
        n_layers = 8
        lookup_len = 5


    else:  # args.size == "xl":
        d_model = 1024
        ff = 2048
        n_heads = 12
        n_layers = 8
        lookup_len = 7

    config = CustomTranslationConfig(
        d_model=d_model,  # Hidden size for T5 Medium
        d_ff=ff,  # Set feed-forward hidden size
        d_kv=d_model // n_heads,  # Set key and value dimension
        decoder_start_token_id=0,  # Set start token id
        dropout_rate=0.1,  # Set dropout rate
        initializer_factor=1.0,  # Set initializer factor
        is_encoder_decoder=True,  # Set model as encoder-decoder
        layer_norm_epsilon=1e-6,  # Set layer norm epsilon
        n_positions=512,  # Set maximum sequence length
        output_past=True,  # Set output past
        relative_attention_num_buckets=32,  # Set number of buckets for relative attention
        num_layers=n_layers,
        meta_type=meta_type,
        lookup_len=lookup_len,
        **kwargs
    )
    return CustomTranslationModel(config)
