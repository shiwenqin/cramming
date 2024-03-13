"""This rewrite is a simplified version of the proposed changes that actually compiles statically in torch 2.0.

This model is the final, optimized crammed model.

Not all ablations discussed in the paper are implemented as switches in this version,
for all those, check scriptable_bert.py on the old branch.

"""
import torch
from transformers import PretrainedConfig, PreTrainedModel

from typing import Optional
from omegaconf import OmegaConf
from pathlib import Path

from .components import (
    _get_norm_fn,
    _get_nonlin_fn,
    EmbeddingComponent,
    PoolingComponent,
    PredictionHeadComponent,
    AttentionComponent,
    FFNComponent,
    get_extended_attention_mask,
    _init_module,
)
from ..utils import find_component_checkpoint

class crammedBertConfig(PretrainedConfig):
    model_type = "crammedBERT"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)


def construct_crammed_bert(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    config = crammedBertConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    config.arch["embedding"]["vocab_size"] = vocab_size
    config.arch["num_labels"] = downstream_classes

    if downstream_classes is None:
        if config.arch["objective_layout"] == "MLM":
            model = ScriptableLMForPreTraining(config)
        else:
            raise ValueError(f"Invalid layout {config.arch['objective_layout']} of training objective given.")
    else:
        model = ScriptableLMForSequenceClassification(config)
    return model


class TransformerLayer(torch.nn.Module):
    """A transformer-encoder structure based on the components from above."""

    def __init__(self, idx, cfg_arch):
        super().__init__()
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)
        self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.attn = AttentionComponent(
            idx,
            cfg_arch.hidden_size,
            cfg_arch.attention,
            cfg_arch.use_bias,
        )
        self.LAYOUT = self.attn.LAYOUT

        self.ffn = FFNComponent(
            cfg_arch.hidden_size,
            cfg_arch.intermed_size,
            _get_nonlin_fn(cfg_arch.nonlin),
            cfg_arch.use_bias,
        )

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):
        states = states + self.dropout(self.attn(self.norm1(states), attention_mask))
        states = states + self.dropout(self.ffn(self.norm2(states)))
        return states


class ScriptableLM(PreTrainedModel):
    """Simplified transformer wrapper."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.embedding = EmbeddingComponent(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)
        self.layers = torch.nn.ModuleList([TransformerLayer(idx, self.cfg) for idx in range(self.cfg.num_transformer_layers)])
        self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.use_causal_attention)
        hidden_states = self.embedding(input_ids)

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, attention_mask)

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        return self.final_norm(hidden_states)


class ScriptableLMForPreTraining(PreTrainedModel):
    """Pretraining version with optional prediction head and variant for sparse prediction."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableLM(config)

        if not self.cfg.skip_head_transform:
            self.prediction_head = PredictionHeadComponent(self.cfg)
        else:
            self.prediction_head = torch.nn.Identity()  # from linear in old version

        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sparse_prediction = self.cfg.sparse_prediction

        self.param_dict = self._get_param_dict()

        self._init_weights()

        if self.cfg.active:
            self.source_archs, self.weight_dicts, self.total_layer_source = find_component_checkpoint(config)
            self._load_weights(self.weight_dicts, self.source_archs, self.total_layer_source, self.cfg.start_layer)


    def _get_param_dict(self):
        param_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_dict[name] = param
        return param_dict

    
    def _load_single_layer(self, weight_dict, layer_num_source, layer_num_target):
        target_prefix = f"encoder.layers.{layer_num_target}."
        source_prefix = f"encoder.layers.{layer_num_source}."
        for name, param in weight_dict.items():
            #print(name, param.shape)
            if param.shape == torch.Size([]):
                continue
            if name.startswith(source_prefix):
                name_source = name.replace(source_prefix, target_prefix)
                self.param_dict[name_source].data = param.data
                print(f"Loaded {name_source} from {name}")


    def _load_weights(self, weight_dicts, archs, total_layer_num, start_layer=0):
        if total_layer_num + start_layer > self.cfg.num_transformer_layers:
            raise ValueError("Too many layers in the combined model to fit into the current model.")
        curr_layer = start_layer
        for i, (weight_dict, arch) in enumerate(zip(weight_dicts, archs)):
            num_layers = arch.num_transformer_layers
            for j in range(num_layers):
                self._load_single_layer(weight_dict, j, j + curr_layer)
            curr_layer += num_layers


    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )
        

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        outputs = self.encoder(input_ids, attention_mask)
        outputs = outputs.view(-1, outputs.shape[-1])

        if self.sparse_prediction and labels is not None:
            masked_lm_loss = self._forward_sparse(outputs, labels)
        else:
            outputs = self.decoder(self.prediction_head(outputs))
            if labels is not None:
                masked_lm_loss = self.loss_fn(outputs, labels.view(-1))
            else:
                masked_lm_loss = outputs.new_zeros((1,))

        return {"loss": masked_lm_loss, "outputs": outputs}

    # Sparse prediction usually has an unpredictable number of entries in each batch
    # but the dataloader was modified so that 25% of the batch is ALWAYS masked.
    # This allows for static compilation. If you modify the dataloader, this function will fill your compile cache
    def _forward_sparse(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):

        labels = labels.view(-1)
        mask_positions = labels.view(-1) != self.loss_fn.ignore_index
        num_masks_guaranteed = round(self.sparse_prediction * labels.shape[0])
        # outputs = outputs[mask_positions]  # not allowed as dynamic shape op
        # labels = labels[mask_positions]
        # torch.masked_select(labels, mask_positions)  # not allowed as a dynamic shape operator

        # indices = torch.arange(mask_positions.shape[0], device=outputs.device)[mask_positions] # not allowed
        indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]  # ugh

        outputs = outputs[indices]  # not allowed as dynamic shape op, but ok with indices
        labels = labels[indices]
        # alternative:
        # outputs = torch.take_along_dim(outputs, indices.view(-1, 1), 0)
        # labels = torch.take(labels, indices)

        outputs = self.decoder(self.prediction_head(outputs))
        masked_lm_loss = self.loss_fn(outputs, labels)
        return masked_lm_loss


class ScriptableLMForSequenceClassification(PreTrainedModel):
    """Classification head and pooler."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.num_labels = self.cfg.num_labels

        self.encoder = ScriptableLM(config)
        self.pooler = PoolingComponent(self.cfg.classification_head, self.cfg.hidden_size)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
        self._init_weights()

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        logits = self.head(self.pooler(self.encoder(input_ids, attention_mask)))

        if labels is not None:
            if self.problem_type is None:  # very much from huggingface
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        else:
            loss = logits.new_zeros((1,))

        return dict(logits=logits, loss=loss)


class ScriptableLMForTokenClassification(PreTrainedModel):
    """Classification head without pooling."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableLM(config)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
        self._init_weights()

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        logits = self.head(self.encoder(input_ids, attention_mask))

        if labels is not None:
            if self.problem_type is None:  # very much from huggingface
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            else:
                raise ValueError("Wrong problem type!")
        else:
            loss = logits.new_zeros((1,))

        return dict(logits=logits, loss=loss)

