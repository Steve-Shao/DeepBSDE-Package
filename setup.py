from setuptools import setup, find_packages

setup(
    name='DeepBSDE',
    description='A deep learning-based solver for high-dimensional PDEs using BSDEs',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'tensorflow==2.13',
        'munch',
        'absl-py'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)