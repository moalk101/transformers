from setuptools import setup, find_packages

setup(
    name='mytransformer',
    version='0.0.1',
    packages=find_packages(where='src'), 
    package_dir={'': 'src'},            
    description='A custom transformer package',
    author='Modar',
    install_requires=[
        
    ],
    python_requires='>=3.6',
)