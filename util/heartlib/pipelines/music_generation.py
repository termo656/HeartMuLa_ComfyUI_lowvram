from tokenizers import Tokenizer
from ..heartmula.modeling_heartmula import HeartMuLa
from ..heartcodec.modeling_heartcodec import HeartCodec
import torch
from typing import Dict, Any, Optional
import os
from dataclasses import dataclass
from tqdm import tqdm
import torchaudio
import json
import gc
import scipy.io.wavfile
from transformers import BitsAndBytesConfig

# !!! ВАЖНО: Импорт для проверки прерывания
import comfy.model_management

@dataclass
class HeartMuLaGenConfig:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0

    @classmethod
    def from_file(cls, path: str):
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        return cls(**data)

class HeartMuLaGenPipeline:
    def __init__(
        self,
        model: Optional[HeartMuLa],
        audio_codec: Optional[HeartCodec],
        muq_mulan: Optional[Any],
        text_tokenizer: Tokenizer,
        config: HeartMuLaGenConfig,
        device: torch.device,
        dtype: torch.dtype,
        heartmula_path: Optional[str] = None,
        heartcodec_path: Optional[str] = None,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        num_quantizers: Optional[int] = None,
    ):
        self.model = model
        self.audio_codec = audio_codec
        self.muq_mulan = muq_mulan
        self.text_tokenizer = text_tokenizer
        self.config = config
        self.device = device
        self.dtype = dtype
        self.heartmula_path = heartmula_path
        self.heartcodec_path = heartcodec_path
        self.bnb_config = bnb_config
        self._parallel_number = num_quantizers + 1 if num_quantizers else 9
        self._muq_dim = model.config.muq_dim if model else None

    def load_heartmula(self):
        if self.model is None:
            print(f"Загрузка HeartMuLa из {self.heartmula_path}...")
            device_map = {"": self.device} if self.bnb_config else None
            
            self.model = HeartMuLa.from_pretrained(
                self.heartmula_path, 
                torch_dtype=self.dtype, 
                quantization_config=self.bnb_config,
                device_map=device_map
            )
            
            if not self.bnb_config:
                self.model.to(self.device)
                
            self.model.eval()
            self._muq_dim = self.model.config.muq_dim
            print("HeartMuLa загружена.")

    def load_heartcodec(self):
        if self.audio_codec is None:
            print(f"Загрузка HeartCodec из {self.heartcodec_path}...")
            try:
                self.audio_codec = HeartCodec.from_pretrained(
                    self.heartcodec_path,
                    torch_dtype=self.dtype 
                )
                self.audio_codec.to(self.device)
                self.audio_codec.eval()
                print("HeartCodec загружен.")
            except Exception as e:
                print(f"Ошибка загрузки HeartCodec: {e}")
                raise e

    def preprocess(self, inputs: Dict[str, Any], cfg_scale: float):
        self.load_heartmula()
        tags = inputs["tags"].lower()
        if not tags.startswith("<tag>"): tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"): tags = f"{tags}</tag>"
        tags_ids = self.text_tokenizer.encode(tags).ids
        if tags_ids[0] != self.config.text_bos_id: tags_ids = [self.config.text_bos_id] + tags_ids
        if tags_ids[-1] != self.config.text_eos_id: tags_ids = tags_ids + [self.config.text_eos_id]

        muq_embed = torch.zeros([self._muq_dim], dtype=self.dtype, device=self.device)
        muq_idx = len(tags_ids)

        lyrics = inputs["lyrics"].lower()
        lyrics_ids = self.text_tokenizer.encode(lyrics).ids
        if lyrics_ids[0] != self.config.text_bos_id: lyrics_ids = [self.config.text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != self.config.text_eos_id: lyrics_ids = lyrics_ids + [self.config.text_eos_id]

        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
        tokens = torch.zeros([prompt_len, self._parallel_number], dtype=torch.long, device=self.device)
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids, device=self.device)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids, device=self.device)
        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool, device=self.device)
        tokens_mask[:, -1] = True

        bs_size = 2 if cfg_scale != 1.0 else 1
        def _cfg_cat(t):
            t = t.unsqueeze(0)
            return torch.cat([t, t], dim=0) if cfg_scale != 1.0 else t

        return {
            "tokens": _cfg_cat(tokens),
            "tokens_mask": _cfg_cat(tokens_mask),
            "muq_embed": _cfg_cat(muq_embed),
            "muq_idx": [muq_idx] * bs_size,
            "pos": _cfg_cat(torch.arange(prompt_len, dtype=torch.long, device=self.device)),
        }

    # Добавлен аргумент progress_callback
    def _forward(self, model_inputs: Dict[str, Any], max_audio_length_ms: int, temperature: float, topk: int, cfg_scale: float, progress_callback=None):
        self.load_heartmula()
        self.model.setup_caches(2 if cfg_scale != 1.0 else 1)
        
        frames = []
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            curr_token = self.model.generate_frame(
                tokens=model_inputs["tokens"], 
                tokens_mask=model_inputs["tokens_mask"], 
                input_pos=model_inputs["pos"], 
                temperature=temperature, 
                topk=topk, 
                cfg_scale=cfg_scale, 
                continuous_segments=model_inputs["muq_embed"], 
                starts=model_inputs["muq_idx"],
            )
        frames.append(curr_token[0:1,])

        total_steps = max_audio_length_ms // 80
        
        # Используем tqdm только для консоли, если не передан колбек
        # Но лучше оставить и то и то
        pbar = tqdm(range(total_steps), desc="Generating audio")
        
        for i in pbar:
            # !!! ПРОВЕРКА НА ОТМЕНУ ГЕНЕРАЦИИ (Кнопка Cancel/Stop)
            comfy.model_management.throw_exception_if_processing_interrupted()
            
            # !!! ОБНОВЛЕНИЕ ПРОГРЕССА В COMFYUI
            if progress_callback:
                progress_callback(i)

            padded_token = (torch.ones((curr_token.shape[0], self._parallel_number), device=self.device, dtype=torch.long) * self.config.empty_id)
            padded_token[:, :-1] = curr_token
            padded_token = padded_token.unsqueeze(1)
            padded_token_mask = torch.ones_like(padded_token, dtype=torch.bool); padded_token_mask[..., -1] = False

            with torch.autocast(device_type="cuda", dtype=self.dtype):
                curr_token = self.model.generate_frame(
                    tokens=padded_token, 
                    tokens_mask=padded_token_mask,
                    input_pos=model_inputs["pos"][..., -1:] + i + 1,
                    temperature=temperature, 
                    topk=topk, 
                    cfg_scale=cfg_scale,
                )
            if torch.any(curr_token[0:1, :] >= self.config.audio_eos_id): break
            frames.append(curr_token[0:1,])
        
        return torch.stack(frames).permute(1, 2, 0).squeeze(0).cpu()

    def postprocess(self, frames: torch.Tensor, save_path: str, keep_model_loaded: bool):
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()

        try:
            self.load_heartcodec()
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    wav = self.audio_codec.detokenize(frames.to(self.device))
                wav = wav.detach().cpu().float()
            
            if wav.dim() == 3 and wav.shape[0] == 1:
                wav = wav.squeeze(0)
            if wav.dim() == 2 and wav.shape[0] < wav.shape[1]:
                wav = wav.permute(1, 0)
            
            try:
                wav_np = wav.numpy()
                scipy.io.wavfile.write(save_path, 48000, wav_np)
                print(f"Аудио сохранено: {save_path}")
            except Exception as e:
                print(f"Ошибка scipy: {e}, пробую torchaudio")
                if wav.shape[0] > wav.shape[1]: wav = wav.permute(1, 0)
                torchaudio.save(save_path, wav, 48000)

        finally:
            if not keep_model_loaded:
                if self.audio_codec is not None:
                    del self.audio_codec
                    self.audio_codec = None
            
            torch.cuda.empty_cache()
            gc.collect()

    def __call__(self, inputs: Dict[str, Any], **kwargs):
        keep_model_loaded = kwargs.get("keep_model_loaded", True)
        
        # Извлекаем колбек прогресса из kwargs
        progress_callback = kwargs.get("progress_callback", None)
        
        model_inputs = self.preprocess(inputs, cfg_scale=kwargs.get("cfg_scale", 1.5))
        
        frames = self._forward(model_inputs, 
                               max_audio_length_ms=kwargs.get("max_audio_length_ms", 120000),
                               temperature=kwargs.get("temperature", 1.0),
                               topk=kwargs.get("topk", 50),
                               cfg_scale=kwargs.get("cfg_scale", 1.5),
                               progress_callback=progress_callback) # Передаем в forward
                               
        self.postprocess(frames, kwargs.get("save_path", "out.wav"), keep_model_loaded)

    @classmethod
    def from_pretrained(cls, pretrained_path: str, device: torch.device, dtype: torch.dtype, version: str, bnb_config=None, lazy_load=True):
        heartcodec_path = os.path.join(pretrained_path, "HeartCodec-oss")
        heartmula_path = os.path.join(pretrained_path, f"HeartMuLa-oss-{version}")
        tokenizer = Tokenizer.from_file(os.path.join(pretrained_path, "tokenizer.json"))
        gen_config = HeartMuLaGenConfig.from_file(os.path.join(pretrained_path, "gen_config.json"))
        return cls(None, None, None, tokenizer, gen_config, device, dtype, heartmula_path, heartcodec_path, bnb_config)
