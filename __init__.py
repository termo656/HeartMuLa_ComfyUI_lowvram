import os
import sys
import torch
import torchaudio
import numpy as np
import uuid
import gc  # 引入 gc 进行垃圾回收
import folder_paths
from transformers import BitsAndBytesConfig
# ----------------------------
# 添加本地 HeartLib 到路径
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.join(current_dir, "util")
if util_dir not in sys.path:
    sys.path.insert(0, util_dir)

# ----------------------------
# 路径配置
# ----------------------------
MODEL_BASE_DIR = os.path.join(folder_paths.models_dir, "HeartMuLa")

# ----------------------------
# 全局模型管理器
# ----------------------------
class HeartMuLaModelManager:
    _instance = None
    _gen_pipes = {}
    _transcribe_pipe = None 
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(HeartMuLaModelManager, cls).__new__(cls)
        return cls._instance

    def get_gen_pipeline(self, version="3B"):
        if version not in self._gen_pipes:
            print(f"[HeartMuLa] 正在加载生成管线 (版本: {version}) 于 {self._device}...")
            from heartlib import HeartMuLaGenPipeline
            from transformers import BitsAndBytesConfig
            
            # 配置 4-bit 量化
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            self._gen_pipes[version] = HeartMuLaGenPipeline.from_pretrained(
                MODEL_BASE_DIR,
                device=self._device,
                dtype=torch.float16,
                version=version,
                bnb_config=bnb_config,
            )
            print(f"[HeartMuLa] 生成管线 ({version}) 就绪。")
            
        return self._gen_pipes[version]

    def get_transcribe_pipeline(self):
        if self._transcribe_pipe is None:
            print(f"[HeartMuLa] 正在加载转录管线 于 {self._device}...")
            from heartlib import HeartTranscriptorPipeline
            
            self._transcribe_pipe = HeartTranscriptorPipeline.from_pretrained(
                MODEL_BASE_DIR,
                device=self._device,
                dtype=torch.float16,
            )
            print("[HeartMuLa] 转录管线 就绪。")
            
        return self._transcribe_pipe

# ----------------------------
# 节点: 音乐生成器
# ----------------------------
class HeartMuLa_Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True, "placeholder": "[Verse]\n..."}),
                "tags": ("STRING", {"multiline": True, "placeholder": "piano,happy,wedding"}),
                "version": (["3B", "7B"], {"default": "3B"}),
                "max_audio_length_ms": ("INT", {"default": 240000, "min": 10000, "max": 600000, "step": 10000}),
                "topk": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio_output", "filepath")
    FUNCTION = "generate"
    CATEGORY = "HeartMuLa"

    def generate(self, lyrics, tags, version, max_audio_length_ms, topk, temperature, cfg_scale):
        manager = HeartMuLaModelManager()
        pipe = manager.get_gen_pipeline(version)

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"heartmula_gen_{uuid.uuid4().hex}.mp3"
        out_path = os.path.join(output_dir, filename)

        # --- 生成 ---
        with torch.no_grad():
            pipe(
                {"lyrics": lyrics, "tags": tags},
                max_audio_length_ms=max_audio_length_ms,
                save_path=out_path,
                topk=topk,
                temperature=temperature,
                cfg_scale=cfg_scale,
            )
        
        # --- 内存清理 ---
        # 生成完成后立即清理内存
        torch.cuda.empty_cache()
        gc.collect()

        # 加载结果回 ComfyUI
        waveform, sample_rate = torchaudio.load(out_path)
        
        print(f"[HeartMuLa Gen] 加载形状: {waveform.shape}")
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) 
        
        # 显式转换为 float32
        waveform = waveform.float()
            
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
            
        print(f"[HeartMuLa Gen] 输出形状: {waveform.shape}")

        audio_output = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

        return (audio_output, out_path)

# ----------------------------
# 节点: 歌词转录器
# ----------------------------
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
        # --- PRE-PROCESSING ---
        if isinstance(audio_input, dict):
            waveform = audio_input["waveform"]
            sr = audio_input["sample_rate"]
        else:
            sr, waveform = audio_input
            if isinstance(waveform, np.ndarray):
                 waveform = torch.from_numpy(waveform)
        
        print(f"[HeartMuLa Transcribe] Input Shape: {waveform.shape}, Dtype: {waveform.dtype}")

        if waveform.ndim == 3:
            waveform = waveform.squeeze(0)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        # --- CRITICAL: Convert to Float32 before saving ---
        if waveform.dtype != torch.float32:
            print("[HeartMuLa Transcribe] Casting to float32 for WAV save...")
            waveform = waveform.float()
        
        print(f"[HeartMuLa Transcribe] Saving Shape: {waveform.shape}, Dtype: {waveform.dtype}")
        
        output_dir = folder_paths.get_temp_directory()
        os.makedirs(output_dir, exist_ok=True)
        temp_filename = f"heartmula_transcribe_in_{uuid.uuid4().hex}.wav"
        temp_path = os.path.join(output_dir, temp_filename)

        torchaudio.save(temp_path, waveform, sr)

        # --- INFERENCE ---
        try:
            temp_tuple = tuple(float(x.strip()) for x in temperature_tuple.split(","))
        except:
            temp_tuple = (0.0, 0.1, 0.2, 0.4)

        manager = HeartMuLaModelManager()
        pipe = manager.get_transcribe_pipeline()

        with torch.no_grad():
            result = pipe(
                temp_path,
                temperature=temp_tuple,
                no_speech_threshold=no_speech_threshold,
                logprob_threshold=logprob_threshold,
                compression_ratio_threshold=1.8,
                max_new_tokens=256,
                num_beams=2,
                task="transcribe",
                condition_on_prev_tokens=False
            )

        # --- MEMORY CLEANUP ---
        # Clear memory immediately after transcription finishes
        torch.cuda.empty_cache()
        gc.collect()

        # --- POST-PROCESSING ---
        if os.path.exists(temp_path):
            os.remove(temp_path)

        text = result if isinstance(result, str) else result.get("text", str(result))
        return (text,)

# ----------------------------
# 节点映射
# ----------------------------
NODE_CLASS_MAPPINGS = {
    "HeartMuLa_Generate": HeartMuLa_Generate,
    "HeartMuLa_Transcribe": HeartMuLa_Transcribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeartMuLa_Generate": "HeartMuLa Music Generator",
    "HeartMuLa_Transcribe": "HeartMuLa Lyrics Transcriber",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']