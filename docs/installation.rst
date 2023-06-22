.. _installation:

============
Installation
============

Code repository: https://github.com/math-a3k/covid-ht

Requirements
============

* A working `Python`_ 3.7 or newer (3.11 recommended) installation so you can issue succesfully its version:

.. code-block:: shell

    > python --version

* The `pipenv`_ package:

.. code-block:: shell

    > python -m pip install pipenv

* A working `git`_ installation so you can so you can issue succesfully its version:

.. code-block:: shell

    > git --version

* Accesible instances of Postgres and Redis servers.

Steps
=====

* Clone the repository:

.. code-block:: shell

    > git clone https://github.com/math-a3k/tradero

* Change into the directory and install the Python requirements:

.. code-block:: shell

    > cd tradero
    > pipenv install


See :ref:`running` for information on executing the program.

.. _Python: https://www.python.org/
.. _pipenv: https://pipenv.pypa.io/en/latest/
.. _git: https://git-scm.com/
