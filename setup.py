import os
import sys
from setuptools import setup, find_packages

dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dirname, "volcano"))


with open(os.path.join(dirname, "requirements.txt"), "r") as f:
    install_requires = f.read().splitlines()


with open(os.path.join(dirname, "README.md"), encoding="utf-8") as f:
    readme = f.read()


setup(
    name="volcano",
    version="0.1.0dev",
    author="Fran Bartolić, Rodrigo Luger, Daniel-Foreman Mackey",
    author_email="fb90@st-andrews.ac.uk",
    url="https://github.com/fbartolic/volcano",
    license="MIT",
    packages=find_packages(),
    description="Mapping Io with Starry",
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    zip_safe=False,
)
