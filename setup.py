from setuptools import setup, find_packages

setup(
    name="medicine_redefine",
    version="0.1.0",
    description="药物重定位项目：寻找已有药物的新用途",
    author="QLi007",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "biopython",
        "rdkit",
        "scikit-learn",
        "torch",
        "requests",
        "jupyter",
    ],
    python_requires=">=3.8",
) 