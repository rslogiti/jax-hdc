<p align="center">
    <a href="https://github.com/yourusername/jax-hdc/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat" /></a>
        <a href="https://pypi.org/project/jax-hdc/"><img alt="pypi version" src="https://img.shields.io/pypi/v/jax-hdc.svg?style=flat&color=blue" /></a>
    <a href="https://github.com/yourusername/jax-hdc/actions/workflows/test.yml?query=branch%3Amain"><img alt="tests status" src="https://img.shields.io/github/actions/workflow/status/yourusername/jax-hdc/test.yml?branch=main&label=tests&style=flat" /></a>
    <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat" />
</p>

<div align="center">
    <a href="https://github.com/yourusername/jax-hdc">
    </a>
</div>

# JAX-HDC

JAX-HDC is a Python library for _Hyperdimensional Computing_ (also known as _Vector Symbolic Architectures_) built on JAX.

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

There are several examples [in the repository](https://github.com/yourusername/jax-hdc/tree/main/examples). Here is a simple one to get you started:

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

Contributions of new models to the library are welcome!

## Acknowledgments

JAX-HDC is inspired by the excellent [TorchHD](https://github.com/hyperdimensional-computing/torchhd) library and aims to bring the power of JAX to the Hyperdimensional Computing community. I thank the TorchHD authors for their foundational work in creating accessible HDC tools.
