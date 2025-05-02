from setuptools import setup, find_packages

setup(
    name="llm_factory",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pytest",
        "transformers",
        "datasets",
    ],
)