#!/bin/bash

set -euo pipefail

# pipenv install
pipenv run python manage.py collectstatic --no-input
pipenv run python manage.py migrate --no-input
pipenv run pytest
pipenv run python manage.py load_symbols
