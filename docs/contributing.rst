Contributing
============

We welcome contributions to JAX-HDC! This guide will help you get started.

Development Setup
-----------------

1. Fork the repository
2. Clone your fork:

.. code-block:: bash

   git clone https://github.com/yourusername/jax-hdc.git
   cd jax-hdc

3. Create a virtual environment:

.. code-block:: bash

   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate

4. Install in development mode:

.. code-block:: bash

   pip install -e ".[dev]"

Code Style
----------

We use the following tools to maintain code quality:

* **Black** for code formatting
* **isort** for import sorting
* **flake8** for linting
* **mypy** for type checking

Run these tools before submitting a pull request:

.. code-block:: bash

   black .
   isort .
   flake8
   mypy jax_hdc

Testing
-------

Run tests with:

.. code-block:: bash

   pytest

Submitting Changes
------------------

1. Create a new branch for your feature
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

Documentation
-------------

Documentation is built using Sphinx. To build the docs locally:

.. code-block:: bash

   cd docs
   make html