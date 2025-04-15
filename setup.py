# setup.py
import os
import sys
import glob
import warnings
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch # Import torch to check CUDA availability

# --- CUDA Availability Check ---
WITH_CUDA = False
if torch.cuda.is_available():
    try:
        # Attempt a simple CUDA operation to be sure
        _ = torch.tensor([1.0, 2.0]).cuda()
        WITH_CUDA = True
        print("CUDA detected. Building with CUDA support.")
    except Exception as e:
        warnings.warn(f"CUDA appears to be available but failed verification: {e}. Building without CUDA.")
else:
    warnings.warn("CUDA not available. Building without CUDA support.")

# --- Source Files ---
cpp_source_dir = os.path.join('ultra_rwka', 'backend', 'cpp')
cuda_source_dir = os.path.join('ultra_rwka', 'backend', 'cuda')
include_dir = os.path.join('ultra_rwka', 'backend', 'include')

cpp_sources = glob.glob(os.path.join(cpp_source_dir, '*.cpp'))
# Optional CPU sources
# cpu_sources = glob.glob(os.path.join(cpp_source_dir, 'cpu', '*.cpp'))
# cpp_sources += cpu_sources

cuda_sources = []
if WITH_CUDA:
    # Find .cu files in subdirectories (fa_kla, wavelets, idam, etc.)
    cuda_sources = glob.glob(os.path.join(cuda_source_dir, '**', '*.cu'), recursive=True)
    if not cuda_sources:
         warnings.warn(f"Building with CUDA, but no .cu files found in {cuda_source_dir}")

# --- Include Directories ---
include_dirs = [
    os.path.abspath(include_dir)
]
# Add PyTorch's include path - important for torch/extension.h
pytorch_include_dir = os.path.dirname(torch.__file__)
include_dirs.append(os.path.join(pytorch_include_dir, 'include'))
include_dirs.append(os.path.join(pytorch_include_dir, 'include', 'torch', 'csrc', 'api', 'include'))


# --- Compiler Arguments ---
extra_compile_args = {'cxx': ['-O3', '-std=c++17']} # Use C++17
if WITH_CUDA:
    extra_compile_args['nvcc'] = ['-O3', '-std=c++17']
    # Add specific GPU architectures if needed, e.g.,
    # extra_compile_args['nvcc'].extend(['-gencode', 'arch=compute_70,code=sm_70'])


# --- Extension Definition ---
extension_sources = cpp_sources + cuda_sources
extension_name = 'ultra_rwka_backend'

# Select Extension type based on CUDA availability
if WITH_CUDA and cuda_sources: # Only build CUDAExtension if .cu files exist
    Extension = CUDAExtension
else:
    Extension = CppExtension
    extension_sources = cpp_sources # Exclude cuda sources if not building with CUDA or none found
    warnings.warn("Building CPU-only extension.")


extensions = [
    Extension(
        name=extension_name,
        sources=[s for s in extension_sources if os.path.exists(s)], # Ensure sources exist
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args
    )
]

# --- Setup ---
# Read requirements
try:
    with open('requirements.txt', 'r') as f:
        install_requires = f.read().splitlines()
except FileNotFoundError:
    print("Warning: requirements.txt not found. Proceeding without package dependencies.")
    install_requires = []

# Read README
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = 'Ultra-RWKA Sequence Modeling Library'


setup(
    name='ultra_rwka',
    version='0.1.0',
    author='Quantvmh', # Assuming from paper
    description='Ultra-RWKA Sequence Modeling Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=('tests*', 'experiments*', 'docs*', 'paper*')),
    ext_modules=extensions,
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True) # Avoid abi suffix for simplicity
    },
    install_requires=install_requires,
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License", # Make sure this matches your LICENSE file
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 3 - Alpha", # Initial development stage
    ],
  
    # url='https://github.com/0xAEC/something',
)
