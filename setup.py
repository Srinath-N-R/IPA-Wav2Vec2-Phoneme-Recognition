from setuptools import setup, find_packages

setup(
    name="aligner",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "datasets",
        "transformers",
        "evaluate",
        "librosa",
        "torch",
        "torchaudio",
        "accelerate",
        "tqdm",
        "pathlib",
        "matplotlib",
        "PyYAML",
        "wandb",
        "jiwer",
        "mpi4py",
        "deepspeed"
    ]
)
