import os
import sys
import torch
import torchaudio
import numpy as np
import uuid
import gc
import folder_paths
import logging
import warnings
from transformers import BitsAndBytesConfig
import comfy.utils

# Оптимизация памяти для PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torchtune").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.join(current_dir, "util")
if util_dir not in sys.path:
    sys.path.insert(0, util_dir)

# ----------------------------
# Path Configuration
# ----------------------------
folder_paths.add_model_folder_path("HeartMuLa", os.path.join(folder_paths.models_dir, "HeartMuLa"))

def get_model_base_dir():
    paths = folder_paths.get_folder_paths("HeartMuLa")
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]

MODEL_BASE_DIR = get_model_base_dir()

class HeartMuLaModelManager:
    _instance = None
    _gen_pipes = {}
    _transcribe_pipe = None 
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HeartMuLaModelManager, cls).__new__(cls)
        return cls._instance

    def get_gen_pipeline(self, version="3B", quantization="none"):
        pipe_key = f"{version}_{quantization}"
        
        if pipe_key in self._gen_pipes:
            return self._gen_pipes[pipe_key]


        if self._gen_pipes:
            print(f"[HeartMuLa] Выгрузка старых моделей перед загрузкой {pipe_key}...")
            keys_to_remove = list(self._gen_pipes.keys())
            for k in keys_to_remove:
                del self._gen_pipes[k]
            torch.cuda.empty_cache()
            gc.collect()

        from heartlib import HeartMuLaGenPipeline
        
        bnb_config = None
        # Настройка квантования
        if quantization == "4bit":
            print(f"[HeartMuLa] Включение 4-bit (NF4) квантования для {version}...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, # Это ключевой параметр для экономии памяти
                bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        elif quantization == "8bit":
            print(f"[HeartMuLa] Включение 8-bit квантования для {version}...")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        print(f"[HeartMuLa] Загрузка модели {version} ({quantization})...")
        self._gen_pipes[pipe_key] = HeartMuLaGenPipeline.from_pretrained(
            MODEL_BASE_DIR,
            device=self._device,
            dtype=torch.float16, 
            version=version,
            bnb_config=bnb_config
        )
        torch.cuda.empty_cache()
        gc.collect()
        
        return self._gen_pipes[pipe_key]

    def get_transcribe_pipeline(self):
        if self._transcribe_pipe is None:
            from heartlib import HeartTranscriptorPipeline
            self._transcribe_pipe = HeartTranscriptorPipeline.from_pretrained(
                MODEL_BASE_DIR, device=self._device, dtype=torch.float16,
            )
            torch.cuda.empty_cache()
            gc.collect()
        return self._transcribe_pipe
    
    # Метод для принудительной очистки
    def unload_all(self):
        self._gen_pipes.clear()
        self._transcribe_pipe = None
        torch.cuda.empty_cache()
        gc.collect()

class HeartMuLa_Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True, "placeholder": "[Verse]\n..."}),
                "tags": ("STRING", {"multiline": True, "placeholder": "Drum and Bass, dub, uplifting..."}),
                "version": (["3B", "7B"], {"default": "3B"}),
                # Убрал 2bit, так как это невозможно
                "quantization": (["none", "4bit", "8bit"], {"default": "4bit"}), 
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_audio_length_ms": ("INT", {"default": 240000, "min": 10000, "max": 600000, "step": 10000}),
                "topk": ("INT", {"default": 50, "min": 1, "max": 250, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio_output", "filepath")
    FUNCTION = "generate"
    CATEGORY = "HeartMuLa"

    def generate(self, lyrics, tags, version, quantization, seed, max_audio_length_ms, topk, temperature, cfg_scale, keep_model_loaded):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)

        manager = HeartMuLaModelManager()
        
        try:
            pipe = manager.get_gen_pipeline(version, quantization)

            output_dir = folder_paths.get_output_directory()
            os.makedirs(output_dir, exist_ok=True)
            filename = f"heartmula_gen_{uuid.uuid4().hex}.wav"
            out_path = os.path.join(output_dir, filename)

            steps = max_audio_length_ms // 80
            pbar = comfy.utils.ProgressBar(steps)

            def update_progress(step):

                pbar.update(1)

            with torch.inference_mode():
                pipe(
                    {"lyrics": lyrics, "tags": tags},
                    max_audio_length_ms=max_audio_length_ms,
                    save_path=out_path,
                    topk=topk,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    keep_model_loaded=keep_model_loaded,
                    progress_callback=update_progress # !!! ПЕРЕДАЕМ КОЛБЕК
                )
        except Exception as e:
            print(f"Generation failed or cancelled: {e}")
            manager.unload_all()
            raise e
        finally:
            if not keep_model_loaded:
                manager.unload_all()
            
            torch.cuda.empty_cache()
            gc.collect()

        if os.path.exists(out_path):
            waveform, sample_rate = torchaudio.load(out_path)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0) 
            waveform = waveform.float()
            if waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)
            audio_output = {"waveform": waveform, "sample_rate": sample_rate}
            return (audio_output, out_path)
        else:
            raise FileNotFoundError("Audio file was not generated (possibly cancelled).")

class HeartMuLa_Transcribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_input": ("AUDIO",),
                "temperature_tuple": ("STRING", {"default": "0.0,0.1,0.2,0.4"}),
                "no_speech_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "logprob_threshold": ("FLOAT", {"default": -1.0, "min": -5.0, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics_text",)
    FUNCTION = "transcribe"
    CATEGORY = "HeartMuLa"

    def transcribe(self, audio_input, temperature_tuple, no_speech_threshold, logprob_threshold):
        if isinstance(audio_input, dict):
            waveform = audio_input["waveform"]
            sr = audio_input["sample_rate"]
        else:
            sr, waveform = audio_input
            if isinstance(waveform, np.ndarray):
                 waveform = torch.from_numpy(waveform)
        
        if waveform.ndim == 3:
            waveform = waveform.squeeze(0)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        waveform = waveform.to(torch.float32).cpu()
        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)
        temp_path = os.path.join(output_dir, f"hm_trans_{uuid.uuid4().hex}.wav")
        torchaudio.save(temp_path, waveform, sr)

        try:
            temp_tuple = tuple(float(x.strip()) for x in temperature_tuple.split(","))
        except:
            temp_tuple = (0.0, 0.1, 0.2, 0.4)

        manager = HeartMuLaModelManager()
        pipe = manager.get_transcribe_pipeline()

        try:
            with torch.inference_mode():
                result = pipe(
                    temp_path,
                    temperature=temp_tuple,
                    no_speech_threshold=no_speech_threshold,
                    logprob_threshold=logprob_threshold,
                    task="transcribe",
                )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # Транскрибатор обычно маленький, но можно тоже чистить
            torch.cuda.empty_cache()
            gc.collect()

        text = result if isinstance(result, str) else result.get("text", str(result))
        return (text,)

NODE_CLASS_MAPPINGS = {
    "HeartMuLa_Generate": HeartMuLa_Generate,
    "HeartMuLa_Transcribe": HeartMuLa_Transcribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeartMuLa_Generate": "HeartMuLa Music Generator",
    "HeartMuLa_Transcribe": "HeartMuLa Lyrics Transcriber",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

