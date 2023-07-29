.. _commands:

========
Commands
========

The following Django commands - meant to be ran as ``python manage.py command_name [--argument [arg_val]]`` in the virtual environment - are bundled:


.. _load_symbols:

``load_symbols``
================

Loads symbols from Binance.

.. autoclass:: base.management.commands.load_symbols.Command

.. _warm_and_ready:

``warm_and_ready``
==================

Cleans current data, gets fresh one and updates models.

.. autoclass:: base.management.commands.warm_and_ready.Command

.. _ws_klines:

``ws_klines``
=============

Starts the websocket client for data retrieval (ws polling).

.. autoclass:: base.management.commands.ws_klines.Command

.. _calibrate_all_models:

``calibrate_all_models``
=========================

Callibrates the parameters of all Symbols' Prediction models. This a computational expensive call.

.. autoclass:: base.management.commands.calibrate_all_models.Command

.. _calibrate_all_windows:

``calibrate_all_windows``
=========================

Callibrates the window size of all Symbols' Prediction models. This a computational expensive call.

.. autoclass:: base.management.commands.calibrate_all_windows.Command
