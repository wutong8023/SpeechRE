import re
import copy
import logging
from typing import Optional
from omegaconf import DictConfig, open_dict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils
from fairseq.tasks import FairseqTask
from fairseq.modules import (
    LayerNorm,
    FairseqDropout,
)
from fairseq.models import register_model
from fairseq.models.transformer import TransformerDecoder
from fairseq.models.wav2vec import (
    Wav2Vec2Seq2SeqConfig,
    Wav2Vec2Seq2SeqModel,
    Wav2VecEncoder,
    Embedding,
)
from fairseq.models.speech_to_text import (
    lengths_to_padding_mask,
    Conv1dSubsampler,
)

logger = logging.getLogger(__name__)

BLOCKS2REGEX = {
    "encoder.feat_extr": r"encoder.*\.feature_extractor\..*|"
                         r"encoder.*\.post_extract_proj\..*|"
                         r"encoder.*\.pos_conv\..*",
    "encoder.self_attn": r"encoder.*\.self_attn\..*",
    "encoder.layer_norm": r"encoder.*layer_norm.*",
    "encoder.ffn": r"encoder.*\.fc[1-2]\..*",
    "adapter": r"encoder\.adapter.*",
    "len_adaptor": r"encoder\.len_adaptor.*",
    "decoder.embedding": r"decoder\.embed_tokens.*|"
                         r"decoder\.embed_positions.*|"
                         r"decoder\.layernorm_embedding.*",
    "decoder.self_attn": r"decoder.*\.self_attn\..*",
    "decoder.layer_norm": r"decoder.*layer_norm.*",
    "decoder.encoder_attn": r"decoder.*\.encoder_attn\..*",
    "decoder.ffn": r"decoder.*\.fc[1-2]\..*",
}


@dataclass
class Wav2Vec2Seq2SeqModConfig(Wav2Vec2Seq2SeqConfig):
    freeze_layers: str = field(
        default="",
        metadata={"help": "finetune only LayerNorm and Attention (LNA) layers"}
    )
    adapter_dim: Optional[int] = field(
        default=None,
        metadata={"help": "projection size of the Adapter"}
    )
    adapter_post: bool = field(
        default=False,
        metadata={"help": "if true, the Adapter is placed after the "
                          "Length Adaptor"}
    )
    adapter_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability for the encoder-decoder "
                          "attention weights (if it's not specified, the "
                          "decoder_attention_dropout is used)"}
    )
    len_adaptor_kernel_sizes: str = field(
        default="3,3",
        metadata={"help": "kernel sizes of the Length Adaptor (Conv1d)"}
    )
    len_adaptor_channels: int = field(
        default=1024,
        metadata={"help": "# of channels in the Length Adaptor (Conv1d)"}
    )
    load_pretrained_decoder_from: Optional[str] = field(
        default=None,
        metadata={"help": "model to take decoder weights from"}
    )
    adapter_input_dim: int = field(
        default=1024,
        metadata={"help": "encoder dimension"}
    )
    adapter_output_dim: int = field(
        default=100,
        metadata={"help": "decoder dimension"}
    )
    decoder_output_dim: int = field(
        default=768,
        metadata={"help": "decoder output dimension (extra linear layer "
                          "if different from decoder embed dim)"}
    )
    decoder_enc_attention_dropout: Optional[float] = field(
        default=None,
        metadata={"help": "dropout probability for the encoder-decoder "
                          "attention weights (if it's not specified, the "
                          "decoder_attention_dropout is used)"}
    )


@register_model("wav2vec_seq2seq_speechre", dataclass=Wav2Vec2Seq2SeqModConfig)
class Wav2Vec2Seq2SeqModModelRE(Wav2Vec2Seq2SeqModel):
    """
    Modified version of the wav2vec_seq2seq model.

    It adds these functionalities:
      - Use with the speech_to_text pipeline
      - Loading pretrained decoder
      - Finetuning only LNA layers
      - Using adapter and length_adaptor modules
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Seq2SeqModConfig, task: FairseqTask):
        """Build a new model instance."""

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(task.tgt_dict, cfg.decoder_embed_dim)
        # print(dir(task.tgt_dict))
        # print(decoder_embed_tokens)

        encoder = cls.build_encoder(cfg)
        decoder = cls.build_decoder(cfg, task.tgt_dict, decoder_embed_tokens)

        model = Wav2Vec2Seq2SeqModModelRE(encoder, decoder)
        model.freeze_blocks(cfg)
        return model

    @classmethod
    def build_encoder(cls, cfg: Wav2Vec2Seq2SeqModConfig):
        return Wav2VecEncoderMod(cfg)

    @classmethod
    def build_text_encoder(cls, cfg: Wav2Vec2Seq2SeqModConfig):
        from fairseq.models.bart import BARTModel
        bart = BARTModel.from_pretrained("/data/wangguitao/IWSLT/Pre-trained_models/BART-large",
                                         checkpoint_file='model.pt')
        bart.eval()
        return bart

    @classmethod
    def build_decoder(cls, cfg: Wav2Vec2Seq2SeqModConfig, tgt_dict, embed_tokens):
        from fairseq.models.bart import BARTModel
        def get_rebel():
            from transformers import AutoModelForSeq2SeqLM


            rebel = AutoModelForSeq2SeqLM.from_pretrained("/data/wangguitao/IWSLT/Pre-trained_models/REBEL-large")
            rebel_decoder = rebel.model.decoder
            bart = BARTModel.from_pretrained("/data/wangguitao/IWSLT/Pre-trained_models/BART-large",
                                             checkpoint_file='model.pt')
            speechre_decoder = bart.model.decoder

            # bart.eval()
            # tokens = bart.encode('Hello world!')
            # last_layer_features = bart.extract_features(tokens)
            # print(last_layer_features.shape)
            # breakpoint()

            speechre_decoder.embed_tokens = Embedding(50268, 1024, padding_idx=1)
            speechre_decoder.embed_tokens.weight.data[:50265, :] = rebel_decoder.embed_tokens.weight.data[:50265, :]
            speechre_decoder.embed_tokens.weight.data[50265, :] = rebel_decoder.embed_tokens.weight.data[50267, :]
            speechre_decoder.embed_tokens.weight.data[50266, :] = rebel_decoder.embed_tokens.weight.data[50266, :]
            speechre_decoder.embed_tokens.weight.data[50267, :] = rebel_decoder.embed_tokens.weight.data[50265, :]
            speechre_decoder.embed_positions.weight.data = rebel_decoder.embed_positions.weight.data
            for i in range(len(speechre_decoder.layers)):
                speechre_decoder.layers[i].self_attn.k_proj = rebel_decoder.layers[i].self_attn.k_proj
                speechre_decoder.layers[i].self_attn.v_proj = rebel_decoder.layers[i].self_attn.v_proj
                speechre_decoder.layers[i].self_attn.q_proj = rebel_decoder.layers[i].self_attn.q_proj
                speechre_decoder.layers[i].self_attn.out_proj = rebel_decoder.layers[i].self_attn.out_proj

                speechre_decoder.layers[i].encoder_attn.k_proj = rebel_decoder.layers[i].encoder_attn.k_proj
                speechre_decoder.layers[i].encoder_attn.v_proj = rebel_decoder.layers[i].encoder_attn.v_proj
                speechre_decoder.layers[i].encoder_attn.q_proj = rebel_decoder.layers[i].encoder_attn.q_proj
                speechre_decoder.layers[i].encoder_attn.out_proj = rebel_decoder.layers[i].encoder_attn.out_proj

                speechre_decoder.layers[i].self_attn_layer_norm = rebel_decoder.layers[i].self_attn_layer_norm
                speechre_decoder.layers[i].encoder_attn_layer_norm = rebel_decoder.layers[i].encoder_attn_layer_norm
                speechre_decoder.layers[i].fc1 = rebel_decoder.layers[i].fc1
                speechre_decoder.layers[i].fc2 = rebel_decoder.layers[i].fc2
                speechre_decoder.layers[i].final_layer_norm = rebel_decoder.layers[i].final_layer_norm
            speechre_decoder.build_output_projection(speechre_decoder.args, tgt_dict, speechre_decoder.embed_tokens)

            del bart
            del rebel

            return speechre_decoder

        # decoder = get_rebel()
        # from fairseq.models.bart import BARTModel
        # bart = BARTModel.from_pretrained(cfg.load_pretrained_decoder_from, checkpoint_file='model.pt')
        # text = "<triplet> AP <subj> NEW YORK <obj> OrgBased_In"
        # tensor([0,
        #         41552, 21237, 26151, 15698,     -> < triplet >
        #         1480,                           ->  AP
        #         28696, 10936, 267, 15698,       ->  < subj >
        #         5178, 4180,                     ->  NEW YORK
        #         28696, 46134, 15698,            ->  < obj >
        #         1793, 571, 20930, 1215, 1121,   ->  OrgBased_In
        #         2])
        # print(text)
        # text = bart.encode(text)
        # print(text)
        # exit(0)
        bart = BARTModel.from_pretrained("/data/wangguitao/IWSLT/Pre-trained_models/BART-large",
                                         checkpoint_file='model.pt')
        decoder = bart.model.decoder
        old_embed = decoder.embed_tokens
        new_embed = embed_tokens    # nn.Embedding
        old_embed_dim = old_embed.weight.data.shape[0]
        new_embed.weight.data[:old_embed_dim, :] = old_embed.weight.data[:old_embed_dim, :]
        decoder.embed_tokens = new_embed
        decoder.build_output_projection(decoder.args, tgt_dict, embed_tokens)

        # decoder = TransformerDecoderMod(cfg, tgt_dict, embed_tokens)
        # if getattr(cfg, "load_pretrained_decoder_from", None):
        #     decoder = checkpoint_utils.load_pretrained_component_from_model(
        #         component=decoder, checkpoint=cfg.load_pretrained_decoder_from
        #     )
        #     logger.info(
        #         f"loaded pretrained decoder from: "
        #         f"{cfg.load_pretrained_decoder_from}"
        #     )
        return decoder

    def forward(self, src_text, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        # print(src_tokens.shape)
        # print(src_tokens)
        # print(src_lengths.shape)
        # print(src_lengths)
        # exit(0)
        encoder_out = self.encoder(
            src_text, src_tokens, src_lengths=src_lengths, **kwargs
        )
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out


    def freeze_blocks(self, cfg: Wav2Vec2Seq2SeqModConfig):
        regex_to_freeze = re.compile(
            "|".join([BLOCKS2REGEX[b] for b in cfg.freeze_layers.split(',')])
        )
        for n, p in self.named_parameters():
            if re.match(regex_to_freeze, n):
                p.requires_grad = False

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs



import torch.nn as nn


from typing import List, Dict, Optional, Any


from transformers.generation_utils import GenerationMixin

# Wav2Triplet from wt
'''
class Wav2Triplet(SpreeModel, GenerationMixin):
    def __init__(self, model_args):
        super(Wav2Triplet, self).__init__()
        self.encoder = Wav2VecEncoder()  # tips: CTC
        self.encoder_config = self.encoder.config
        self.encoder_processor = self.encoder.processor

        self.decoder = BartDecoder()
        self.decoder_config = self.decoder.config
        self.decoder_tokenizer = self.decoder.tokenizer

        # FIXME: used for self.generate(), but the config does not exactly matched with architecture.
        self.config = self.decoder_config

        self.adapter = LengthAdapter(adapter_input_dim=self.encoder_config.hidden_size,
                                     adapter_output_dim=self.decoder_config.hidden_size,
                                     adapter_hidden_size=200)

    def forward(self, *args, **kwargs):
        # encoder
        input_values = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        encoder_output = self.encoder(input_values=input_values, attention_mask=attention_mask)

        encoder_hidden_states = encoder_output.hidden_states
        encoder_attentions = encoder_output.attentions

        # adapter
        encoder_hidden_states = tuple(
            [self.adapter(encoder_hidden_state) for encoder_hidden_state in encoder_hidden_states])
        # decoder
        # decoder_input = kwargs["decoder_input"]
        labels = kwargs["labels"]

        print('start decoding......')  # DEBUG

        decoder_output = self.decoder(  ######### decoder这一步出问题了，报错了
            input_ids=labels,
            attention_mask=attention_mask,  # 新加的
            encoder_hidden_states=encoder_hidden_states,
            encoder_attentions=encoder_attentions
        )
        print('end decoding......')
        print(f'decoder_output = {decoder_output}')  # DEBUG

        # logits
        logits = self.decoder.decoder_head(decoder_output[0]) + self.decoder.decoder_final_logits_bias
        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        output = {
            "loss": masked_lm_loss,
            "logits": logits,
            "encoder_output": encoder_output,
            "decoder_output": decoder_output,
        }

        print(f"output = {output}")  # DEBUG
        return output

    def collate_fn(self, batch_data: List[Instance]):
        input_audios = []
        input_text = []
        target_sequences = []
        for instance in batch_data:
            input_audios.append(instance.audio_array)
            input_text.append(instance.text)  # 这个哪里来的？
            target_sequences.append(instance.linearization)

        # prepare for encoder
        encoder_input = self.encoder_processor(input_audios, sampling_rate=16000, return_tensors='pt',
                                               padding='longest', return_attention_mask=True)
        print(f'encoder_input.keys() = encoder_input.keys()')

        # prepare for decoder
        decoder_input = self.decoder_tokenizer(target_sequences, return_tensors='pt', padding='longest')

        # wrapped as a dict
        wrapped_batch = {
            "input_ids": encoder_input.input_values,  # Named as input_ids for API matching.
            "attention_mask": encoder_input.attention_mask,
            "labels": decoder_input.input_ids,
        }

        print(f'wrapped_batch = {wrapped_batch}')
        return wrapped_batch
'''

class Wav2VecEncoderMod(Wav2VecEncoder):
    """
    Modification of the Wav2VecEncoder

    It modifies it to work with the speech_to_text pipeline.
    Moreover, it includes the adapter and length adaptor modules.
    """
    def __init__(self, cfg: Wav2Vec2Seq2SeqModConfig, tgt_dict=None):
        super().__init__(cfg, tgt_dict)
        self.len_adaptor = Conv1dSubsampler(
            cfg.decoder_embed_dim,
            cfg.len_adaptor_channels,
            cfg.decoder_embed_dim,
            [int(k) for k in cfg.len_adaptor_kernel_sizes.split(",")],
        )
        # from fairseq.models.bart import BARTModel
        # self.bart = BARTModel.from_pretrained("/data/wangguitao/IWSLT/Pre-trained_models/BART-large",
        #                                  checkpoint_file='model.pt')
        # self.bart.eval()

    def forward(self, src_text, src_tokens, src_lengths, **kwargs):
        encoder_out = super().forward(
            source=src_tokens,
            padding_mask=lengths_to_padding_mask(src_lengths),
            tbc=False,
            **kwargs
        )
        encoder_out["encoder_padding_mask"] = encoder_out.pop("padding_mask")
        encoder_out["encoder_out"], lengths = self.len_adaptor(
            encoder_out["encoder_out"],
            (~encoder_out["encoder_padding_mask"]).sum(dim=1)
        )
        encoder_out["encoder_padding_mask"] = lengths_to_padding_mask(lengths)
        # for k, v in encoder_out.items():
        #     print(k)
        #     print(v.shape)
        #     print(v)
        # pad = nn.ZeroPad2d((0, 0, 0, 1024 - encoder_out["encoder_out"].shape[1]))
        # encoder_out["encoder_out"] = pad(encoder_out["encoder_out"])
        # for k, v in encoder_out.items():
        #     print(k)
        #     print(v.shape)
        #     print(v)
        # encoder_out["encoder_padding_mask"] = encoder_out.pop("padding_mask")
        # encoder_out["encoder_out"] = self.len_adapter(encoder_out["encoder_out"])
        # encoder_out["encoder_padding_mask"] = lengths_to_padding_mask(
        #     torch.tensor([encoder_out["encoder_out"].shape[0]] * encoder_out["encoder_out"].shape[1]).cuda()
        # )
        # encoder_out["padding_mask"] = encoder_out["encoder_padding_mask"].transpose(0, 1)
        # for k, v in encoder_out.items():
        #     print(k)
        #     print(v.shape)
        #     print(v)
        return {k: [v] for k, v in encoder_out.items()}

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out, new_order):
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
        }


# Wav2VecEncoder from wt
'''
class Wav2VecEncoder(nn.Module):
    """
    base encoder: Wav2Vec2
    """

    def __init__(self, *args, **kwargs):
        super(Wav2VecEncoder, self).__init__()
        self.encoder = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h',
                                                      cache_dir='../cached_models/')
        self.config = self.encoder.config
        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h',
                                                           cache_dir='../cached_models/')
        if "sampling_rate" in kwargs:
            self.sampling_rate = kwargs["sampling_rate"]
        else:
            self.sampling_rate = 16000

    def forward(self, **kwargs):
        """
        encode raw audio
        Args:
            encoder_input

        Returns:
        output (Wav2Vec2BaseModelOutput)
        """
        output = self.encoder(output_hidden_states=True, output_attentions=True, **kwargs)
        return output
'''


class TransformerDecoderMod(TransformerDecoder):
    """
    Modification of the TransformerDecoder

    It is adapted to the argument names defined in Wav2Vec2Seq2SeqModConfig.
    """
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False):
        transformer_cfg = copy.deepcopy(cfg)
        with open_dict(transformer_cfg):
            transformer_cfg.dropout = transformer_cfg.decoder_dropout
            transformer_cfg.attention_dropout = (
                transformer_cfg.decoder_attention_dropout
            )
            transformer_cfg.activation_dropout = (
                transformer_cfg.decoder_activation_dropout
            )
            transformer_cfg.layernorm_embedding = True
            transformer_cfg.adaptive_input = False
            transformer_cfg.no_scale_embedding = False
            transformer_cfg.quant_noise_pq = 0.0
            transformer_cfg.adaptive_softmax_cutoff = None
        super().__init__(transformer_cfg, dictionary, embed_tokens, no_encoder_attn)
        if cfg.decoder_enc_attention_dropout is not None:
            for layer in self.layers:
                layer.encoder_attn.dropout_module.p = \
                    cfg.decoder_enc_attention_dropout

    def load_state_dict(self, state_dict, strict=True):
        state_dict["output_projection.weight"] = state_dict["embed_tokens.weight"]
        super().load_state_dict(state_dict, strict)


# BartDecoder from wt
'''
class BartDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BartDecoder, self).__init__()
        self.lm = BartForConditionalGeneration.from_pretrained('facebook/bart-base', cache_dir='../cached_models')
        self.config = self.lm.config
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', cache_dir="../cached_models")

        # add special token
        self.num_new_tokens = self.tokenizer.add_tokens(['<subj>', '<obj>', "<triplet>"])
        # self.lm.resize_token_embeddings(self.num_new_tokens)
        self.lm.resize_token_embeddings(10000)  # 这里改了
        self.decoder = deepcopy(self.lm.get_decoder())
        self.decoder_head = deepcopy(self.lm.lm_head)
        self.decoder_final_logits_bias = deepcopy(self.lm.final_logits_bias)

        del self.lm

        self.modify_architecture()  # if necessary
        pass

    def forward(self, *args, **kwargs):
        decoder_outputs = self.decoder(
            input_ids=kwargs["input_ids"],  #
            attention_mask=kwargs["attention_mask"],  #
            encoder_hidden_states=kwargs["encoder_hidden_states"][-1],  #
            encoder_attention_mask=kwargs["encoder_attentions"],  #
            head_mask=None,  #
            cross_attn_head_mask=None,
            past_key_values=None,
            inputs_embeds=None,  # important, used for generation
            use_cache=None,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        return decoder_outputs

    def modify_architecture(self):
        """
        modify the architecture of bart decoder -- insert adapter
        Returns:
        None
        """
        pass

    @classmethod
    def shift_tokens_right(cls, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    @classmethod
    def prepare_decoder_input_ids_from_labels(cls, labels: torch.Tensor):
        return cls.shift_tokens_right(labels,
                                      cls.config.pad_token_id,
                                      cls.config.decoder_start_token_id)
'''


class Adapter(nn.Module):
    """
    Adapter for model finetuning, as described in:
    https://arxiv.org/pdf/1909.08478.pdf
    """
    def __init__(self, embed_dim, proj_dim, dropout):
        super().__init__()
        self.layer_norm = LayerNorm(embed_dim)
        self.down_proj = nn.Linear(embed_dim, proj_dim)
        self.up_proj = nn.Linear(proj_dim, embed_dim)
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = F.relu(x)
        x = self.up_proj(x)
        x = self.dropout_module(x)
        x += residual
        return x


# LengthAdapter from wt
class LengthAdapter(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LengthAdapter, self).__init__()
        self.input_dim = kwargs["adapter_input_dim"]
        self.output_dim = kwargs["adapter_output_dim"]
        self.mid_dim = kwargs["adapter_hidden_size"]
        self.adapter = nn.Sequential(
            nn.Linear(self.input_dim, self.mid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.mid_dim),
            nn.Linear(self.mid_dim, self.output_dim)
        )
        pass

    def forward(self, batch_data):
        """
        transfer input data

        Args:
            batch_data (Tensor): the hidden states of encoder

        Returns:
        tensor
        """
        bsz, in_seq_len, _ = batch_data.size()  # B x T x (C x D)
        x = batch_data.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        x = self.adapter(x)
        output = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return output
