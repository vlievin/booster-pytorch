import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='booster-pytorch',
    version='0.0.1',
    author="Valentin Lievin",
    author_email="valentin.lievin@gmail.com",
    description="A lightweight library to ease the training and the debugging of deep neural networks with PyTorch. Data structures and paradigms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vlievin/booster-pytorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch',
        'tqdm',
        'numpy',
        'matplotlib'
    ],
)
