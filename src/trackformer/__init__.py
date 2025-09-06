import os

# Ajout des DLL nécessaires à PyTorch et CUDA pour Windows
torch_lib_path = os.path.join(os.environ["CONDA_PREFIX"], "Lib", "site-packages", "torch", "lib")
cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin"

if os.path.isdir(torch_lib_path):
    os.add_dll_directory(torch_lib_path)
if os.path.isdir(cuda_bin_path):
    os.add_dll_directory(cuda_bin_path)
