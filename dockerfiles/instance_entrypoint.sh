#!/bin/bash

set -euo pipefail

pipenv run python manage.py collectstatic --no-input
pipenv run python manage.py migrate --no-input
pipenv run pytest -v
pipenv run python manage.py load_symbols

pipenv run supervisord -c ./supervisor.conf --nodaemon
