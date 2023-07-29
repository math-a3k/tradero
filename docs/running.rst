.. _running:

==========================
Running a tradero instance
==========================

Docker
======

The easiest way of running an instance is by using `Docker`_:

.. code-block:: shell

    > docker-compose up

The instance should be available at http://localhost in your machine.

You may create a superuser with:

.. code-block:: shell

    > docker exec -it tradero-instance-1 python manage.py createsuperuser

Please refer to its documentation for information on installing it on your system.

Regular
=======

``tradero`` is a "regular" `Django`_ application (``tradero.settings``) which requires a `Postgres`_ database server and a `Redis`_ server properly configured to run.

Activate the project's virtual environment:

.. code-block:: shell

    > pipenv shell

You may check everything is in place by running the testing suite:

.. code-block:: shell

    > pytest

Run the migrations and optionally create a superuser:


.. code-block:: shell

    > pytthon manage.py migrate
    > pytthon manage.py createsuperuser  # optional

After all the previous are set, review ``tradero/settings.py`` and the instance can be executed with:

.. code-block:: shell

    > python manage.py runserver && celery -A tradero beat && celery -A tradero worker -l INFO


.. _Docker: https://www.docker.com/
.. _Django: https://www.djangoproject.com/
.. _Postgres: https://www.postgresql.org/
.. _Redis: https://redis.io/
