<p align="center">
    <a href="https://github.com/yourusername/jax-hdc/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat" /></a>
    <a href="https://github.com/yourusername/jax-hdc/actions/workflows/test.yml?query=branch%3Amain"><img alt="tests status" src="https://img.shields.io/github/actions/workflow/status/yourusername/jax-hdc/test.yml?branch=main&label=tests&style=flat" /></a>
    <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat" />
</p>

<div align="center">
    <a href="https://github.com/yourusername/jax-hdc">
        <img width="380px" alt="JAX-HDC logo" src="https://raw.githubusercontent.com/yourusername/jax-hdc/main/docs/images/jax-hdc-logo.svg" />
    </a>
</div>

# JAX-HDC

JAX-HDC is a Python library for _Hyperdimensional Computing_ (also known as _Vector Symbolic Architectures_) built on JAX.

- **Blazingly Fast:** JAX-HDC leverages JAX's just-in-time compilation and vectorization capabilities to deliver state-of-the-art performance for HDC operations. Experience unprecedented speed with automatic hardware acceleration on CPUs, GPUs, and TPUs.
- **Functionally Pure:** Built with JAX's functional programming paradigm, JAX-HDC operations are pure functions that enable seamless composition, transformation, and optimization of your HDC pipelines.
- **Easy-to-use:** JAX-HDC makes it effortless to develop a wide range of Hyperdimensional Computing applications. For newcomers, we provide Pythonic abstractions and examples to get you started fast. For experienced researchers, the modular design gives you unlimited flexibility to prototype novel ideas rapidly.
- **Differentiable:** Harness JAX's automatic differentiation to optimize HDC models end-to-end with gradient-based methods, opening new possibilities for learnable HDC architectures.

## Installation

JAX-HDC will be hosted on [PyPI](https://pypi.org/project/jax-hdc/). First, install JAX using their [installation instructions](https://jax.readthedocs.io/en/latest/installation.html). Then, use one of the following commands to install JAX-HDC:

```bash
pip install jax-hdc
```

For development installation:

```bash
git clone https://github.com/yourusername/jax-hdc.git
cd jax-hdc
pip install -e .
```

## Documentation

You can find documentation for JAX-HDC [on the website](https://jax-hdc.readthedocs.io).

Check out the [Getting Started](https://jax-hdc.readthedocs.io/en/stable/quickstart.html) page for a quick overview.

The API documentation is divided into several sections:

- [`jax_hdc`](https://jax-hdc.readthedocs.io/en/stable/jax_hdc.html)
- [`jax_hdc.embeddings`](https://jax-hdc.readthedocs.io/en/stable/embeddings.html)
- [`jax_hdc.structures`](https://jax-hdc.readthedocs.io/en/stable/structures.html)
- [`jax_hdc.models`](https://jax-hdc.readthedocs.io/en/stable/models.html)
- [`jax_hdc.memory`](https://jax-hdc.readthedocs.io/en/stable/memory.html)
- [`jax_hdc.datasets`](https://jax-hdc.readthedocs.io/en/stable/datasets.html)

You can improve the documentation by sending pull requests to this repository.

## Examples

We have several examples [in the repository](https://github.com/yourusername/jax-hdc/tree/main/examples). Here is a simple one to get you started:

```python
import jax
import jax.numpy as jnp
import jax_hdc

d = 10000  # number of dimensions
key = jax.random.PRNGKey(42)

# create the hypervectors for each symbol
keys_key, values_key = jax.random.split(key)
keys = jax_hdc.random(keys_key, (3, d))
country, capital, currency = keys

usa, mex = jax_hdc.random(values_key, (2, d))  # United States and Mexico
wdc, mxc = jax_hdc.random(values_key, (2, d))  # Washington D.C. and Mexico City
usd, mxn = jax_hdc.random(values_key, (2, d))  # US Dollar and Mexican Peso

# create country representations
us_values = jnp.stack([usa, wdc, usd])
us = jax_hdc.hash_table(keys, us_values)

mx_values = jnp.stack([mex, mxc, mxn])
mx = jax_hdc.hash_table(keys, mx_values)

# combine all the associated information
mx_us = jax_hdc.bind(jax_hdc.inverse(us), mx)

# query for the dollar of mexico
usd_of_mex = jax_hdc.bind(mx_us, usd)

memory = jnp.concatenate([keys, us_values, mx_values], axis=0)
similarities = jax_hdc.cosine_similarity(usd_of_mex, memory)
print(similarities)
# Array([-0.0062,  0.0123, -0.0057, -0.0019, -0.0084, -0.0078,  0.0102,  0.0057,  0.3292])
# The hypervector for the Mexican Peso is the most similar.
```

This example demonstrates JAX-HDC's functional approach to the classic Kanerva example from [What We Mean When We Say "What's the Dollar of Mexico?": Prototypes and Mapping in Concept Space](https://redwood.berkeley.edu/wp-content/uploads/2020/05/kanerva2010what.pdf). Notice how we use JAX's random key system for reproducible randomness and JAX NumPy arrays for efficient computation.

## Supported HDC/VSA Models

Currently, the library supports the following HDC/VSA models:

- [Multiply-Add-Permute (MAP)](https://jax-hdc.readthedocs.io/en/stable/generated/jax_hdc.MAP.html)
- [Binary Spatter Codes (BSC)](https://jax-hdc.readthedocs.io/en/stable/generated/jax_hdc.BSC.html)
- [Holographic Reduced Representations (HRR)](https://jax-hdc.readthedocs.io/en/stable/generated/jax_hdc.HRR.html)
- [Fourier Holographic Reduced Representations (FHRR)](https://jax-hdc.readthedocs.io/en/stable/generated/jax_hdc.FHRR.html)
- [Binary Sparse Block Codes (B-SBC)](https://jax-hdc.readthedocs.io/en/stable/generated/jax_hdc.BSBC.html)
- [Cyclic Group Representation (CGR)](https://jax-hdc.readthedocs.io/en/stable/generated/jax_hdc.CGR.html)
- [Modular Composite Representation (MCR)](https://jax-hdc.readthedocs.io/en/stable/generated/jax_hdc.MCR.html)
- [Vector-Derived Transformation Binding (VTB)](https://jax-hdc.readthedocs.io/en/stable/generated/jax_hdc.VTB.html)

We welcome anyone to help with contributing more models to the library!

## Why JAX for HDC?

JAX brings several compelling advantages for Hyperdimensional Computing:

- **Performance**: JIT compilation and XLA optimization deliver superior performance compared to eager execution frameworks
- **Hardware Acceleration**: Seamless scaling from CPU to GPU to TPU without code changes
- **Functional Programming**: Pure functions enable better composition and transformation of HDC operations
- **Automatic Differentiation**: Enable gradient-based optimization of HDC models and end-to-end differentiable HDC pipelines
- **Vectorization**: `vmap` allows easy batching and parallel processing of HDC operations
- **Parallelization**: `pmap` enables effortless multi-device computation for large-scale HDC applications

## Contributing

We are always looking for people that want to contribute to the library. If you are considering contributing for the first time we acknowledge that this can be daunting, but fear not! You can look through the [open issues](https://github.com/yourusername/jax-hdc/issues) for inspiration on the kind of problems you can work on. If you are a researcher and want to contribute your work to the library, feel free to open a new issue so we can discuss the best strategy for integrating your work.

### Documentation

To build the documentation locally do the following:

1. Use `pip install -r docs/requirements.txt` to install the required packages.
2. Use `sphinx-build -b html docs build` to generate the html documentation in the `/build` directory.

To create a clean build, remove the `/build` and `/docs/generated` directories.

### Creating a New Release

1. Increment the version number in [version.py](https://github.com/yourusername/jax-hdc/blob/main/jax_hdc/version.py) using [semantic versioning](https://semver.org).
2. Create a new GitHub release. Set the tag according to [PEP 440](https://peps.python.org/pep-0440/), e.g., v1.5.2, and provide a clear description of the changes. You can use GitHub's "auto-generate release notes" button. Look at previous releases for examples.
3. A GitHub release triggers a GitHub action that builds the library and publishes it to PyPI in addition to the documentation website.

### Running Tests

To run the unit tests located in [`jax_hdc/tests`](https://github.com/yourusername/jax-hdc/tree/main/jax_hdc/tests) do the following:

1. Use `pip install -r dev-requirements.txt` to install the required development packages.
2. Then run the tests using just `pytest`.

Optionally, to measure the code coverage use `coverage run -m --omit="jax_hdc/tests/**" pytest` to create the coverage report. You can then view this report with `coverage report`.

### License

This library is [MIT licensed](https://github.com/yourusername/jax-hdc/blob/main/LICENSE).

## Cite

If you use JAX-HDC in your work, please cite:

```bibtex
@software{jax_hdc,
  author = {Your Name},
  title = {JAX-HDC: High-Performance Hyperdimensional Computing with JAX},
  url = {https://github.com/yourusername/jax-hdc},
  year = {2024}
}
```

## Acknowledgments

JAX-HDC is inspired by the excellent [TorchHD](https://github.com/hyperdimensional-computing/torchhd) library and aims to bring the power of JAX to the Hyperdimensional Computing community. We thank the TorchHD authors for their foundational work in creating accessible HDC tools.
