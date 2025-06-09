from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from .configuration_RoPEWhisper import WhisperConfig
from .modeling_RoPEWhisper import WhisperForConditionalGeneration
from ..model.spectrogram2 import MelSpectrogram
from ..tokenizer import Tokenizer, EventType
torch.backends.cudnn.benchmark = True

LABEL_IGNORE_ID = -100

                
def get_backbone_model(args, tokenizer: Tokenizer):
    if args.model.name.startswith("google/t5"):
        config = T5Config.from_pretrained(args.model.name)
    elif args.model.name.startswith("openai/whisper"):
        config = WhisperConfig.from_pretrained(args.model.name)
    else:
        raise NotImplementedError

    config.vocab_size = tokenizer.vocab_size_out

    if hasattr(args.model, "overwrite"):
        for k, v in args.model.overwrite.items():
            assert hasattr(config, k), f"config does not have attribute {k}"
            setattr(config, k, v)

    if hasattr(args.model, "add_config"):
        for k, v in args.model.add_config.items():
            assert not hasattr(config, k), f"config already has attribute {k}"
            setattr(config, k, v)

    if args.model.name.startswith("google/t5"):
        model = T5ForConditionalGeneration(config)
    elif args.model.name.startswith("openai/whisper"):
        config.num_mel_bins = args.model.spectrogram.n_mels + 384
        config.gradient_checkpointing = False
        config.rope_type = 'dynamic'
        config.rope_encoder_scaling_factor = 1.0
        config.rope_decoder_scaling_factor = 1.0
        config.pad_token_id = tokenizer.pad_id
        config.bos_token_id = tokenizer.sos_id
        config.eos_token_id = tokenizer.eos_id
        config.max_source_positions = args.data.src_seq_len // 2
        config.max_target_positions = args.data.tgt_seq_len
        model = WhisperForConditionalGeneration(config)
    else:
        raise NotImplementedError
    
    return model, config.d_model, config


class OsuT(nn.Module):
    __slots__ = ["spectrogram", "decoder_embedder", "transformer", "num_classes", "config", "input_features", "loss_fn"]

    def __init__(self, args: DictConfig, tokenizer: Tokenizer):
        super().__init__()

        self.transformer, d_model, self.config = get_backbone_model(args, tokenizer)
        self.num_classes = tokenizer.num_classes
        self.input_features = args.model.input_features
        self.num_mappers = tokenizer.num_mapper_classes
        self.decoder_embedder = nn.Embedding(tokenizer.vocab_size_in, d_model)
        self.decoder_embedder.weight.data.normal_(mean=0.0, std=0.02)

        self.spectrogram = MelSpectrogram(
            args.model.spectrogram.sample_rate, args.model.spectrogram.n_fft,
            args.model.spectrogram.n_mels, args.model.spectrogram.hop_length
        )


        self.song_pos_embedder = SongPositionEmbedder(
            hidden_size=128,
            num_basis=10
        )

        self.difficulty_embedder = DifficultyEmbedder(
            hidden_size=128,
            max_difficulty=10.0
        )

        
        self.mapper_embedder = MapperStyleEmbedder(
            num_mappers=self.num_mappers,
            embedding_dim=128,
        )

        self.vocab_size_out = tokenizer.vocab_size_out
        class_weights = torch.ones(self.vocab_size_out)
        class_weights[tokenizer.event_start[EventType.TIME_SHIFT]:tokenizer.event_end[EventType.TIME_SHIFT]] = args.data.rhythm_weight

        self.loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="none",
            ignore_index=LABEL_IGNORE_ID,
            label_smoothing=0.1
        )

    def forward(
            self,
            frames: Optional[torch.FloatTensor] = None,
            global_pos: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.Tensor] = None,
            difficulty: Optional[torch.Tensor] = None,
            mapper_idx: Optional[torch.LongTensor] = None,
            encoder_outputs: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            sample_weights: Optional[torch.FloatTensor] = None,
            **kwargs
    ) -> Seq2SeqLMOutput:
        """
        frames: B x L_encoder x mel_bins, float32
        decoder_input_ids: B x L_decoder, int64
        beatmap_idx: B, int64
        beatmap_id: B, int64
        encoder_outputs: B x L_encoder x D, float32
        """

        if encoder_outputs is None: 
            frames_spec = self.spectrogram(frames)  
            input_features = torch.swapaxes(frames_spec, 1, 2) 

            pos_embedding = self.song_pos_embedder(global_pos) 
            pos_embedding_expanded = pos_embedding.unsqueeze(2).repeat(1, 1, input_features.shape[2])

            diff_embedding = self.difficulty_embedder(difficulty)  
            diff_embedding_expanded = diff_embedding.unsqueeze(2).repeat(1, 1, input_features.shape[2])

            mapper_embedding = self.mapper_embedder(mapper_idx)  
            mapper_embedding_expanded = mapper_embedding.unsqueeze(2).repeat(1, 1, input_features.shape[2])
            
            input_features = torch.cat((input_features, diff_embedding_expanded, mapper_embedding_expanded, pos_embedding_expanded), dim=1)
            audio_encoder = self.transformer.get_encoder()
            encoder_outputs = audio_encoder( 
                input_features=input_features,
                return_dict=True, 
            )

        decoder_inputs_embeds = self.decoder_embedder(decoder_input_ids)

        output = self.transformer.forward(
            decoder_input_ids=None, 
            decoder_inputs_embeds=decoder_inputs_embeds,
            encoder_outputs=encoder_outputs,
            labels=labels,
            **kwargs 
        )

        if labels is not None:
            unreduced_loss = self.loss_fn(torch.swapaxes(output.logits, 1, -1), labels)
            if sample_weights is not None:
                unreduced_loss *= sample_weights.unsqueeze(1)
            output.loss = unreduced_loss.sum() / (labels != LABEL_IGNORE_ID).sum()

        return output



class SongPositionEmbedder(nn.Module):
    """
    Generates an embedding vector representing the global position and duration
    context of an audio chunk within a larger song.

    It takes a normalized start and end position of an audio chunk
    (e.g., [0.25, 0.30] meaning the chunk covers 25% to 30% of the total song)

    This allows the model to be aware of:
    - Where the current audio chunk begins within the song.
    - Where the current audio chunk ends within the song.
    - Implicitly, the duration or extent of the chunk relative to the song.
    This information can help the model make decisions appropriate for different
    song sections (e.g., intro, verse, chorus, outro) and varying chunk lengths.
    """
    def __init__(self, hidden_size=64, num_basis=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_basis = num_basis
        
        # Learnable basis centers from 0 to 1
        self.register_parameter(
            'basis_centers',
            nn.Parameter(torch.linspace(0, 1, num_basis))
        )
        
        # Learnable basis widths
        self.register_parameter(
            'basis_widths',
            nn.Parameter(torch.ones(num_basis) * 0.1)
        )
        
        self.position_proj = nn.Sequential(
            nn.Linear(num_basis * 2, hidden_size * 2),  # start and end positions 
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),  # Reduce back to original hidden size
            nn.LayerNorm(hidden_size),
        )

        
        # Initialize weights
        for m in self.position_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def compute_basis_functions(self, position):
        # Compute RBF basis functions
        position_expanded = position.unsqueeze(-1)  # [B, 1]
        centers = self.basis_centers.view(1, -1)    # [1, N]
        widths = self.basis_widths.view(1, -1)      # [1, N]
        
        # Gaussian RBF
        basis = torch.exp(
            -(position_expanded - centers).pow(2) / (2 * widths.pow(2))
        )
        return basis
    
    def forward(self, position_range):
        """
        Args:
            position_range: Tensor of shape [B, 2] containing normalized start and end positions
                            position_range[:, 0] is the start position (0 to 1)
                            position_range[:, 1] is the end position (0 to 1)
        """
        # Split start and end positions
        start_pos = position_range[:, 0]
        end_pos = position_range[:, 1]
        
        # Compute basis functions for both positions
        start_basis = self.compute_basis_functions(start_pos)  # [B, num_basis]
        end_basis = self.compute_basis_functions(end_pos)     # [B, num_basis]
        
        # Concatenate bases
        combined_basis = torch.cat([start_basis, end_basis], dim=1)  # [B, num_basis*2]
        
        # Project to embedding space
        return self.position_proj(combined_basis)
    
class DifficultyEmbedder(nn.Module):
    def __init__(self, hidden_size=64, max_difficulty=10.0, num_basis=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_difficulty = max_difficulty
        self.num_basis = num_basis
        
        # Learnable basis centers
        self.register_parameter(
            'basis_centers', 
            nn.Parameter(torch.linspace(0, 1, num_basis))
        )
        
        # Learnable basis widths
        self.register_parameter(
            'basis_widths',
            nn.Parameter(torch.ones(num_basis) * 0.1)
        )
        
        self.difficulty_proj = nn.Sequential(
            nn.Linear(num_basis, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),  
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        
        # Initialize with smaller weights
        for m in self.difficulty_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)  # Reduce gain
                nn.init.zeros_(m.bias)
                
    def compute_basis_functions(self, diff_normalized):
        # Compute RBF basis functions
        diff_expanded = diff_normalized.unsqueeze(-1)  # [B, 1]
        centers = self.basis_centers.view(1, -1)       # [1, N]
        widths = self.basis_widths.view(1, -1)        # [1, N]
        
        # Gaussian RBF
        basis = torch.exp(
            -(diff_expanded - centers).pow(2) / (2 * widths.pow(2))
        )
        return basis
    
    def forward(self, difficulty):
        # Normalize difficulty
        diff_normalized = difficulty / self.max_difficulty
        
        # Compute basis functions
        basis = self.compute_basis_functions(diff_normalized)
        
        # Project to embedding space
        return self.difficulty_proj(basis)
    
    
class MapperStyleEmbedder(nn.Module):
    """
    Embedding layer for mapper styles
    """
    def __init__(self, num_mappers: int, embedding_dim: int = 64, dropout_prob: float = 0.1):
        """
        Args:
            num_mappers: Total number of unique mappers.
            embedding_dim: Size of the embedding vector.
            dropout_prob: Dropout probability for regularization.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_mappers = num_mappers
        
        # Embedding table: num_mappers rows for actual mappers + 1 row for default style (-1)
        self.embedding = nn.Embedding(num_embeddings=num_mappers + 1, embedding_dim=embedding_dim)
        
        # Initialize embeddings with small random values
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        # Dropout for regularization to help small mappers generalize
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Layer normalization to stabilize embeddings (especially for mappers with few maps)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, mapper_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            mapper_ids: Tensor of shape [B] with mapper IDs (long/int), where:
                - IDs >= 0 correspond to specific mappers (0 to num_mappers-1).
                - ID = -1 triggers the default style.

        Returns:
            Embedding tensor of shape [B, embedding_dim] if mapper_ids is provided,
        """
        if mapper_ids is None:
            return None  # No conditioning applied
        
        # Map -1 to the last index (default style) and ensure IDs are valid
        mapper_ids = torch.where(
            mapper_ids == -1,
            torch.tensor(self.num_mappers, device=mapper_ids.device),
            mapper_ids
        )
        
        # Ensure mapper_ids are within bounds (0 to num_mappers)
        mapper_ids = torch.clamp(mapper_ids, min=0, max=self.num_mappers)
        
        # Retrieve embeddings: [B] -> [B, embedding_dim]
        embeddings = self.embedding(mapper_ids)
        
        # Apply dropout and normalization
        embeddings = self.dropout(embeddings)
        embeddings = self.layer_norm(embeddings)
        
        return embeddings