import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Union, Tuple, List
from transformers.cache_utils import Cache


import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.query = nn.Linear(input_dim, 1, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        # attention_mask: (batch_size, seq_len)
        weights = self.query(x).squeeze(-1)  # (batch_size, seq_len)
        if attention_mask is not None:
            # Replace masked positions with a very large negative number
            weights = weights.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(weights, dim=1)  # attention over seq_len
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch_size, input_dim)
        return pooled

import torch
import torch.nn as nn

class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        # attention_mask: (batch_size, seq_len)
        if attention_mask is None:
            return x.mean(dim=1)

        # Expand mask to match x's shape: (batch_size, seq_len, input_dim)
        mask = attention_mask.unsqueeze(-1).type_as(x)  # same dtype as x
        masked_x = x * mask
        sum_x = masked_x.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-8)  # avoid division by zero
        mean_x = sum_x / lengths
        return mean_x



class LlamaWithEmbeddings(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        embedding_dim: int = 768,
        pooling_type: Optional[str] = "mean",  # or "attention"
    ):
        super().__init__(config)
        self.pooling_type = pooling_type
        if pooling_type == "mean":
            self.embedding_pooler = MeanPooling()
        elif pooling_type == "attention":
            self.embedding_pooler = AttentionPooling(embedding_dim, dtype=config.torch_dtype)
        elif pooling_type == "sequence":
            self.embedding_pooler = None
            # this means we will not pool the embeddings
        else:
            raise ValueError(f"Unknown pooling_type: {pooling_type}")
        self.embedding_projection = nn.Linear(embedding_dim, config.hidden_size, dtype=config.torch_dtype)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        embeddings: Optional[torch.FloatTensor] = None,
        embeddings_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get input embeddings
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            inputs_embeds = self.model.embed_tokens(input_ids)

        # if past_key_values is not None and isinstance(past_key_values, Cache):
        #     for i, layer in enumerate(past_key_values):
        #         print(f"Pre Layer {i}: key shape {layer[0].shape}, value shape {layer[1].shape}")
        # print("Pre Cahche position:", cache_position if cache_position is not None else "None")
        # print("Pre Len position ids", len(position_ids) if position_ids is not None else "None")
        # print("Pre Inputs embeds shape:", inputs_embeds.shape)
        # print("Pre Attention mask shape:", attention_mask.shape if attention_mask is not None else "None")

        # Process embeddings if provided
        if embeddings is not None:
            embeddings = embeddings.to(inputs_embeds.dtype)  # Ensure same dtype as inputs_embeds
            if self.embedding_pooler is not None:
                # Pool if embeddings are a sequence, e.g., (batch_size, seq_len_emb, raw_embedding_dim)
                # pooled_embeddings will be (batch_size, raw_embedding_dim)
                pooled_embeddings = self.embedding_pooler(embeddings, attention_mask=embeddings_attention_mask)
            else:
                # Assume embeddings are already (batch_size, raw_embedding_dim)
                pooled_embeddings = embeddings
            
            projected_embeddings = self.embedding_projection(pooled_embeddings)

            # Prepend the embeddings at the beginning of the sequence
            # projected_embeddings.unsqueeze(1) makes it (batch_size, 1, hidden_size)
            if self.pooling_type != "sequence":
                assert len(projected_embeddings.shape) == 2
                projected_embeddings = projected_embeddings.unsqueeze(1)
            else:
                assert len(projected_embeddings.shape) == 3
            inputs_embeds = torch.cat([projected_embeddings, inputs_embeds], dim=1)
            
            # Adjust attention mask if provided
            if attention_mask is not None:
                # Add a 1 at the beginning for the embedding token
                if self.pooling_type != "sequence":
                    embeddings_attention_mask = torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                attention_mask = torch.cat([embeddings_attention_mask, attention_mask], dim=1)
            else:
                raise NotImplementedError("Attention mask is not provided. Please provide it.")

            # Adjust labels if provided
            if labels is not None:
                # Prepend IGNORE_INDEX to labels to align with the new token
                # IGNORE_INDEX is typically -100 for CrossEntropyLoss
                # TODO explore prefix tuning
                ignore_padding = torch.full(
                    (labels.shape[0], projected_embeddings.shape[1]),
                    -100, # -100 is the default ignore index for CrossEntropyLoss
                    dtype=labels.dtype,
                    device=labels.device
                )
                labels = torch.cat([ignore_padding, labels], dim=1)

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.size(0), -1)
        else:
            # Shift position ids if you added tokens
            position_ids = position_ids + projected_embeddings.shape[1]


        if embeddings is not None:
            past_key_values = None  # invalidate cache if embeddings are added

        # print("Post Len position ids", len(position_ids) if position_ids is not None else "None")
        # print("Post Inputs embeds shape:", inputs_embeds.shape)
        # print("Post Attention mask shape:", attention_mask.shape if attention_mask is not None else "None")

        # Get the base model outputs with modified input embeddings
        # If embeddings were prepended, input_ids should be None to avoid conflict,
        # as inputs_embeds is now the primary source of sequence information.
        outputs = self.model(
            input_ids=None, # Always use inputs_embeds if it's prepared
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        
        # Compute logits
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
