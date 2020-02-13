from distutils.core import setup, Extension

module = Extension(
    'MnistIO',
    sources=['MnistIO_module.c'],
    include_dirs=[
        "/home/dario/anaconda3/envs/TF-MLP/include/python3.7m",
        "/home/dario/anaconda3/envs/TF-MLP/lib/python3.7/site-packages/numpy/core/include"
    ]
)

setup(name='MnistIO', version='1.0', description='Provides a C extension for loading the MNIST dataset', ext_modules=[module])