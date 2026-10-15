from setuptools import setup, Extension
import pybind11
import os

# Get the paths from the environment
conda_prefix = os.environ['CONDA_PREFIX']
lib_dir = os.path.join(conda_prefix, 'lib')
include_dir = os.path.join(conda_prefix, 'include')
eigen_include_dir = os.path.join(include_dir, 'eigen3')

# Define the extension module
extension_mod = Extension(
    "coal_openmp_wrapper",
    sources=["coal_parallel.cpp"],
    library_dirs=[lib_dir],
    libraries=[
        'coal', 'boost_filesystem', 'qhull_r', 'octomap', 'octomath', 'assimp', 'stdc++', 'gcc_s', 'pthread', 'm', 'rt', 'c'
    ],
    include_dirs=[include_dir, eigen_include_dir, pybind11.get_include()],
    extra_compile_args=["-fopenmp", "-std=c++17"],  # OpenMP and C++17 support (coal 3.0 + boost::math require C++17)
    extra_link_args=[f"-L{lib_dir}", "-lcoal", "-fopenmp"],
    language="c++"
)

# Setup function
setup(
    name="coal_openmp_wrapper",
    version="0.1",
    ext_modules=[extension_mod],
    install_requires=["pybind11"]
)