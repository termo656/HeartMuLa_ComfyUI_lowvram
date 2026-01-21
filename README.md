Experimental optimization for older 8GB cards (GTX 1070) [4bit,8bit]

I was able to run it on the old card!

Sometimes style tags don't work, possibly due to model compression.

# CFG_SCALE = 5.0    !!!


# HeartMuLa_ComfyUI
ComfyUI Custom Node for HeartMuLa AI Music Generation and Transcript Text

**HeartMuLa** official GITHUB
https://github.com/HeartMuLa/heartlib


How To Use this In Basic: https://youtu.be/F9LFAeUbBIs



<img width="846" height="518" alt="Screenshot_4" src="https://github.com/user-attachments/assets/2d8bfa16-80fa-4c05-83fe-aef3fd95f4da" />

------------------------------------------------------------

# Installation

------------------------------------------------------------

**Step 1**

To reduce the likelihood of errors, perform a clean installation ComfyUI !!!

https://github.com/Comfy-Org/ComfyUI/releases/download/v0.10.0/ComfyUI_windows_portable_nvidia_cu126.7z

...

Install Comfy Manager, file in ComfyUI folder and run

https://github.com/ltdrdata/ComfyUI-Manager/raw/main/scripts/install-manager-for-portable-version.bat

...

Unzip the node into a folder ComfyUI\custom_nodes\HeartMuLa_ComfyUI

https://github.com/termo656/HeartMuLa_ComfyUI_lowvram/archive/refs/heads/main.zip

**Step 2**

cd /HeartMuLa_ComfyUI

Go to the "python_embeded" folder

Type "cmd" in the address bar

And install it with the command

python.exe -m pip install -r ..\ComfyUI\custom_nodes\HeartMuLa_ComfyUI\requirements.txt


**Step 3**

create a bat file to launch run_nvidia_gpu_lowvram.bat

______________________________________________________

@echo off

setlocal

set CUDA_VISIBLE_DEVICES=0

set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

.\python_embeded\python.exe -s ComfyUI\main.py --lowvram --windows-standalone-build

pause

------------------------------------------------------------
**Step 4**

Download ffmpeg

extract the "bin" folder and install it in the system PATH

https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full-shared.7z

# For File structure

------------------------------------------------------------

<img width="1179" height="345" alt="image" src="https://github.com/user-attachments/assets/5087e10e-9815-48ff-bbb4-3a21dc1e54d1" />


------------------------------------------------------------

# Download model files

------------------------------------------------------------
Go to ComfyUI/models 

Use HuggingFace Cli download model weights.

type :

hf download HeartMuLa/HeartMuLaGen --local-dir ./HeartMuLa

hf download benjiaiplayground/HeartMuLa-oss-3B-bf16 --local-dir ./HeartMuLa/HeartMuLa-oss-3B

hf download benjiaiplayground/HeartCodec-oss-bf16 --local-dir ./HeartMuLa/HeartCodec-oss

------------------------------------------------------------

# For Model File structure

------------------------------------------------------------


<img width="1391" height="320" alt="image" src="https://github.com/user-attachments/assets/3b48ff70-2a4f-4f8d-aed2-d0fbc76bb31f" />



------------------------------------------------------------


Model Sources
------------------------------------------------------------

Github Repo: https://github.com/HeartMuLa/heartlib

Paper: https://arxiv.org/abs/2601.10547

Demo: https://heartmula.github.io/

HeartMuLa-oss-3B: https://huggingface.co/benjiaiplayground/HeartMuLa-oss-3B-bf16

HeartCodec-oss: https://huggingface.co/benjiaiplayground/HeartCodec-oss-bf16








Credits
------------------------------------------------------------
HeartMuLa: https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B






