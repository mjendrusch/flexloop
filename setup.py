import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="flexloop",
  version="0.0.1",
  author="Michael Jendrusch",
  author_email="michael.jendrusch@embl.de",
  description="Minimalistic extensible training loop.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/mjendrusch/flexloop/",
  packages=setuptools.find_packages(),
  install_requires=[
    "torch>=2.0",
    "numpy",
    "optax @ git+https://github.com/google-deepmind/optax.git",
    "dm-haiku @ git+https://github.com/deepmind/dm-haiku"
  ],
  classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
  ],
)
