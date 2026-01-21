@echo off
setlocal
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
.\python_embeded\python.exe -s ComfyUI\main.py --lowvram --windows-standalone-build 
pause
