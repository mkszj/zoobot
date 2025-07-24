import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="zoobot",
    version="2.9.0",
    author="Mike Walmsley",
    author_email="walmsleymk1@gmail.com",
    description="Galaxy morphology classifiers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mwalmsley/zoobot",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.9",

    # fiddly
    # extras_require={'foo': ['bar']},  # for testing only
    # extras_require={'pytorch-cu128': ['torchvision']},


        # the cuda versions, for most users

        # the recommended/default version for most users

            # 





    extras_require={

        'pytorch': [  
            'torch >= 2.7.0',  # no longer requires cuda option
            'torchvision',
            'torchaudio',
            'torchmetrics',
            'lightning >= 2.2.5',
            'litmodels',
            'pyro-ppl >= 1.8.6',
            'timm >= 1.0.15'
        ],

        'pytorch-colab': [
            # colab includes everything else pytorch-y already
            # (and reinstalling it will cause problems)
            'lightning >= 2.2.5'
        ],

        # TODO may add narval/Digital Research Canada config

        'utilities': [
            'seaborn',  # for nice plots
            'boto3',    # for AWs s3 access
            'python-dateutil == 2.8.1',  # for boto3  
        ],
        'docs': [
            'Sphinx',
            'sphinxcontrib-napoleon',
            'furo',
            'docutils<0.18',
            'sphinxemoji'
        ]

        # exactly as above, but _cu121 for cuda 12.8 (the current default)
        # 'pytorch-cu121': [
        #     'torch >= 2.3.1+cu121',  # older torch version
        #     'torchvision',
        #     'torchaudio',
        #     'torchmetrics',
        #     'lightning >= 2.2.5',
        #     'albumentations < 2.0.0',
        #     'pyro-ppl >= 1.8.6',
        #     'timm >= 1.0.15'
        # ],     

        # # as above for cuda 11.8
        # # for GPU, you will also need e.g. cudatoolkit=11.3, 11.6
        # # https://pytorch.org/get-started/previous-versions/#v1121
        # 'pytorch-cu118': [
        #     'torch >= 2.3.1+cu118',   # older torch version
        #     'torchvision',
        #     'torchaudio',
        #     'torchmetrics',
        #     'lightning >= 2.2.5',
        #     'albumentations < 2.0.0',
        #     'pyro-ppl >= 1.8.6',
        #     'timm >= 1.0.15'
        # ],  

        # non-cuda options

        # 'pytorch-cpu': [
        #     # A100 GPU currently only seems to support cuda 11.3 on manchester cluster, let's stick with this version for now
        #     # very latest version wants cuda 11.6
        #     'torch >= 2.3.1+cpu',
        #     'torchvision',
        #     'torchaudio',
        #     'torchmetrics',
        #     'lightning >= 2.2.5',
        #     # 'simplejpeg',
        #     'albumentations < 2.0.0',
        #     'pyro-ppl >= 1.8.6',
        #     'timm >= 1.0.15'
        # ],
        # 'pytorch-m1': [
        #     # as above but without the +cpu (and the extra-index-url in readme has no effect)
        #     # all matching pytorch versions for an m1 system will be cpu
        #     'torch >= 2.7.0',
        #     'torchvision',
        #     'torchaudio',
        #     'torchmetrics',
        #     'lightning >= 2.2.5',
        #     'albumentations < 2.0.0',
        #     'pyro-ppl >= 1.8.6',
        #     'timm >= 1.0.15'
        # ],

    },
    install_requires=[
        'h5py',
        'tqdm',
        'pillow',
        'numpy',
        'pandas',
        'scipy',
        'astropy',  # for reading fits
        'scikit-learn >= 1.0.2',
        'matplotlib',
        'pyarrow',  # to read parquet, which is very handy for big datasets
        # for saving metrics to weights&biases (cloud service, free within limits)
        'wandb',
        'webdataset',  # for reading webdataset files
        'huggingface_hub',  # login may be required
        'setuptools',  # no longer pinned
        'galaxy-datasets>=0.0.25'  # for dataset loading (see github/mwalmsley/galaxy-datasets)
    ]
)
