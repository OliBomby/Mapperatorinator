import torch
from transformers import EncoderDecoderCache, Cache, StaticCache

from osuT5.osuT5.model import Mapperatorinator


class MapperatorinatorCache(EncoderDecoderCache):
    def __init__(self, self_attention_cache: Cache, cross_attention_cache: Cache, cfg_scale: float):
        super().__init__(self_attention_cache, cross_attention_cache)
        self.cfg_scale = cfg_scale
        self.is_compileable = False  # https://github.com/huggingface/transformers/pull/38244

    def get_max_cache_shape(self):
        """Returns the maximum shape of the cache."""
        return self.self_attention_cache.get_max_cache_shape()

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        beam_idx = beam_idx.repeat(2) if self.cfg_scale > 1 else beam_idx
        self.self_attention_cache.reorder_cache(beam_idx)
        self.cross_attention_cache.reorder_cache(beam_idx)


def get_cache(model: Mapperatorinator, batch_size: int, num_beams: int = 1, cfg_scale: float = 1.0):
    cache_kwargs = {
        "config": model.config,
        "max_batch_size": batch_size * num_beams * 2 if cfg_scale > 1 else batch_size * num_beams,
        "max_cache_len": model.config.max_target_positions,
        "device": model.device,
        "dtype": model.dtype,
    }
    decoder_cache = StaticCache(**cache_kwargs)
    encoder_kwargs = cache_kwargs.copy()
    encoder_kwargs["max_cache_len"] = model.config.max_source_positions
    encoder_cache = StaticCache(**encoder_kwargs)
    return MapperatorinatorCache(decoder_cache, encoder_cache, cfg_scale)
