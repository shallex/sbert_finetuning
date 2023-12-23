from setuptools import setup, find_packages


setup(
    name="sbert_finetuning",
    version="1.0",
    author="a.sharshavin",
    packages=["sbert_finetuning"],
    package_dir={"": "src"},
    include_package_data=True,
)