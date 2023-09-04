.. _running:

==========================
Running a tradero instance
==========================

Docker
======

The easiest way of running an instance is by using `Docker`_:

.. code-block:: shell

    > docker-compose up

The instance should be available at http://localhost on your machine.

You may create a superuser with:

.. code-block:: shell

    > docker exec -it tradero-instance-1 python manage.py createsuperuser

Please refer to its documentation for information on installing it on your system.

Regular
=======

``tradero`` is a "regular" `Django`_ application (``tradero.settings``) that requires a `Postgres`_ database server and a `Redis`_ server properly configured to run.

Activate the project's virtual environment:

.. code-block:: shell

    > pipenv shell

You may check everything is in place by running the testing suite:

.. code-block:: shell

    > pytest

Run the migrations and optionally create a superuser:


.. code-block:: shell

    > python manage.py migrate
    > python manage.py createsuperuser  # optional

After all the previous are set, review ``tradero/settings.py`` and the instance can be executed with:

.. code-block:: shell

    > python manage.py runserver && celery -A tradero beat && celery -A tradero worker -l INFO


Running in the Cloud
====================

There are several reasons for running the instance in the cloud, perhaps the most notorious are high availability and location independency.

``tradero`` is "cloud-ready" - meaning deployable and scalelable out-of-the-box in a cloud environment.

For a container-based deployment, the main ``docker-compose.yml`` provides a setup where the workers for both Symbols and Bots run on separate images, allowing to scale them vertically (higher threads instances) and horizontally (more worker instances) in cloud environments.

As a reference, a "bare" Django deployment with all the services needed (non-Dockerized) on a single VPS instance - like ``ecs.c5.large`` (2 threads, 4GB RAM) - can handle about 300 symbols and 300 bots *without prediction enabled*.

It may be more expensive than the electrical bill for running on the hardware you may already own (laptop, desktop), *provided that you ensure the availability*.

.. _Docker: https://www.docker.com/
.. _Django: https://www.djangoproject.com/
.. _Postgres: https://www.postgresql.org/
.. _Redis: https://redis.io/
