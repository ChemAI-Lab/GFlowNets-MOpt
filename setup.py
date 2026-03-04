from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


setup(
    name="gflow_vqe",
    version="0.1.0",
    description=(
        "Discrete flow-based generative models for measurement optimization in quantum computing."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    author="Isaac Huidobro",
    author_email="huidobri@mcmaster.ca",
    license="MIT",
    license_files=("LICENSE",),
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "matplotlib",
        "networkx",
        "numpy",
        "openfermion",
        "PennyLane",
        "pyscf",
        "tequila-basic",
        "torch",
        "torch-geometric",
        "tqdm",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
