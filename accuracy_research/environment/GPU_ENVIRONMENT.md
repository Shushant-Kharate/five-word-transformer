# Isolated GPU Environment

The copied baseline environments (`myenv` and `.venv`) remain unchanged.

Windows Application Control blocked CUDA DLLs when the environment was placed
under the Documents workspace. The working accuracy environment is therefore a
separate sibling of the canonical project:

`C:\Users\Shushant\Desktop\five_word_accuracy_gpu_env`

It was created from CPython 3.11. PyTorch is installed from the official CUDA
12.8 wheel index because the machine has an NVIDIA GeForce RTX 5070 Laptop GPU
and a CUDA 13.1-capable driver. CUDA 12.8 wheels are supported by the newer
driver and include an `sm_120` build for this GPU.

Creation commands:

```powershell
$gpuEnv = "C:\Users\Shushant\Desktop\five_word_accuracy_gpu_env"
py -3.11 -m venv $gpuEnv
& "$gpuEnv\Scripts\python.exe" -m pip install --upgrade pip
& "$gpuEnv\Scripts\python.exe" -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
& "$gpuEnv\Scripts\python.exe" -m pip install -r .\accuracy_research\environment\requirements_gpu.txt
```

Verified environment:

- PyTorch: `2.11.0+cu128`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 5070 Laptop GPU`
- Compute capability: `12.0`
- Compiled architecture: `sm_120`
- Baseline checkpoint load and 32-record GPU decode: passed
- Peak memory during baseline decode smoke test: 26.4 MiB
