import torch
import torch.nn as nn
from torch.functional import F

import transformers as ts
import datasets as ds
from datasets import Dataset

from transformers.modeling_outputs import *

import numpy as np

class BottleneckAdapter(nn.Module):
  def __init__(self, dim, reduction_factor=16, activation=nn.GELU()):
    super().__init__()

    self.block = nn.Sequential(
        nn.Linear(dim, dim//reduction_factor),
        activation,
        nn.Linear(dim//reduction_factor, dim)
    )

    self.norm = nn.LayerNorm(dim, eps=1e-12, elementwise_affine=True)

  def forward(self, inputs):
    return self.norm(self.block(inputs) + inputs)

class ResidualWrapper(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, inputs):
    return self.model(inputs) + inputs


class MiniAlbertModel(ts.BertPreTrainedModel):
    def __init__(self, config: ts.PretrainedConfig, pretrained_embeddings=None):
        super().__init__(config)

        self.config = config

        self.activation = nn.GELU()
        self.albert = ts.AlbertModel(config=config)

        self.albert.encoder.embedding_hidden_mapping_in = None

        if pretrained_embeddings != None:
          self.albert.embeddings = pretrained_embeddings

        if config.embedding_size != config.hidden_size:
          self.projector = nn.Linear(config.embedding_size, config.hidden_size)
          self.use_embedding_projection = True
        else:
          self.use_embedding_projection = False

        if config.use_adapter:
          self.att_adapters = nn.ModuleList([BottleneckAdapter(config.hidden_size, config.reduction_factor) for _ in range(config.num_hidden_layers)])
          self.mlp_adapters = nn.ModuleList([BottleneckAdapter(config.hidden_size, config.reduction_factor) for _ in range(config.num_hidden_layers)])

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=None,
    ):

        att = self.albert.encoder.albert_layer_groups[0].albert_layers[0].attention

        mlp = nn.Sequential(
            ResidualWrapper(
                nn.Sequential(
                    self.albert.encoder.albert_layer_groups[0].albert_layers[0].ffn,
                    self.albert.encoder.albert_layer_groups[0].albert_layers[0].activation,
                    self.albert.encoder.albert_layer_groups[0].albert_layers[0].ffn_output,
                    self.albert.encoder.albert_layer_groups[0].albert_layers[0].dropout,)
            ),
            self.albert.encoder.albert_layer_groups[0].albert_layers[0].full_layer_layer_norm,
        )
        
        embedding = self.albert.embeddings

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        attention_maps = []
        hidden_states = []

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape, device=device)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embeddings = embedding(input_ids=input_ids,
                               token_type_ids=token_type_ids)
        
        if self.use_embedding_projection:
          embeddings = self.projector(embeddings)

        hidden_states.append(embeddings)

        output = embeddings

        if self.config.use_adapter:
          for att_adapter, mlp_adapter in zip(self.att_adapters, self.mlp_adapters):
            x = att_adapter(output)
            x, attention_map = att(x,
                                  attention_mask=extended_attention_mask,
                                  output_attentions=True)
            x = mlp_adapter(x)
            x = mlp(x)
            output = x

            hidden_states.append(output)
            attention_maps.append(attention_map)
        else:
          for _ in range(self.config.num_hidden_layers):
            x, attention_map = att(output,
                                  attention_mask=extended_attention_mask,
                                  output_attentions=True)
            x = mlp(x)

            output = x

            hidden_states.append(output)
            attention_maps.append(attention_map)

        return BaseModelOutput(
            last_hidden_state=output,
            hidden_states=hidden_states,
            attentions=attention_maps,
        )


    def trainAdaptersOnly(self):
      if self.config.use_adapter == False:
        self.config.use_adapter = True
        self.att_adapters = nn.ModuleList([BottleneckAdapter(self.config.hidden_size, self.config.reduction_factor) for _ in range(self.config.num_hidden_layers)])
        self.mlp_adapters = nn.ModuleList([BottleneckAdapter(self.config.hidden_size, self.config.reduction_factor) for _ in range(self.config.num_hidden_layers)])

      if self.use_embedding_projection:
        for param in self.projector.parameters():
          param.requires_grad = False

      for param in self.albert.parameters():
        param.requires_grad = False

class MiniAlbertForMaskedLM(ts.BertPreTrainedModel):
    def __init__(self, config: ts.PretrainedConfig, pretrained_embeddings=None):
        super().__init__(config)

        self.albert = MiniAlbertModel(config, pretrained_embeddings)

        self.vocab_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.vocab_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.vocab_projector = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

        self.mlm_loss_fct = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=None,
    ):

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        outputs = self.albert(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states,
                              return_dict=return_dict)

        sequence_output = outputs.last_hidden_state

        prediction_logits = self.vocab_transform(sequence_output)  # (bs, seq_length, dim)
        prediction_logits = self.albert.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        mlm_loss = None
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))

        return MaskedLMOutput(
            loss=mlm_loss,
            logits=prediction_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def trainAdaptersOnly(self):
      self.albert.trainAdaptersOnly()

class MiniAlbertForSequenceClassification(ts.BertPreTrainedModel):
    def __init__(self, config: ts.PretrainedConfig, pretrained_embeddings=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.albert = MiniAlbertModel(config , pretrained_embeddings)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[0][:, 0, :]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def trainAdaptersOnly(self):
        self.albert.trainAdaptersOnly()


class MiniAlbertForTokenClassification(ts.BertPreTrainedModel):
    def __init__(self, config: ts.PretrainedConfig, pretrained_embeddings=None):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.albert = MiniAlbertModel(config , pretrained_embeddings)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def trainAdaptersOnly(self):
        self.albert.trainAdaptersOnly()
