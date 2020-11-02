import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kge-alkis", # Replace with your own username
    version="0.0.1",
    author="Alkis Papadopoulos",
    description="Knowledge graph embeddings package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alkis-pap/kge/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'llvmlite==0.33.0',
        'numpy',
        'scipy',
        'torch',
        'numba',
        'pandas'
    ]
)