.. _settings:

========
Settings
========

.. contents::
    :local:
    :depth: 1

Here's a list of settings specific to ``tradero``, for Django settings refer `here <https://docs.djangoproject.com/en/dev/ref/settings/>`_.

All can be set with environmental variables with the ``TRADERO_`` prefix (i.e. ``TRADERO_QUOTE_ASSET`` for :setting:`QUOTE_ASSET`).

.. setting:: QUOTE_ASSET

``QUOTE_ASSET``
===============

Default: ``BUSD``

The Quote Asset for the Symbols to be retrieved.

.. setting:: TIME_INTERVAL

``TIME_INTERVAL``
=================

Default: ``5``

The Time Resolution *in minutes*  to be used for the data, models, and indicators.

.. setting::SYNC_EXECUTION

``SYNC_EXECUTION``
==================

Default: ``False``

Synchronous Execution of Threads (No Threads). Meant only to be used when running tests.

.. setting:: EXECUTOR_THREADS

``EXECUTOR_THREADS``
====================

Default: ``None``

Amount of Threads to be used when parallelizing code. A way of limiting CPU core usage. It corresponds to the ``max_workers`` parameter of the
`ThreadPoolExecutor <https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor>`_.

.. setting:: USE_TASKS

``USE_TASKS``
=============

Default: ``False``

Use tasks instead of threads for scheduling. A task for updating each Symbol or Bot will be submitted to the workers.

.. setting:: SYMBOLS_QUANTITY

``SYMBOLS_QUANTITY``
====================

Default: ``cpu_count() * 4``

Amount of Symbols to be tracked once they have been ranked by model score and filtered by market size.

.. setting:: WARM_UP_PERIODS

``WARM_UP_PERIODS``
===================

Default: ``1``

Number of periods of data to be retrieved by the ``warm_and_ready`` command. Each period corrresponds to 1000 time intervals (i.e. 2 periods with a time interval of 5 correspond to the data of the last 10000 minutes - 2000 k-lines of 5 minutes).

.. setting:: MARKET_SIZE_THRESHOLD

``MARKET_SIZE_THRESHOLD``
=========================

Default: ``180000``

Minimum Last 24h Volume of Quote Asset of a Symbol. Symbols below this threshold will not be tracked and listed to the User.

.. setting:: MODEL_SCORE_THRESHOLD

``MODEL_SCORE_THRESHOLD``
=========================

Default: ``0.3``

Minimum model score of a Symbol. Symbols below this threshold will not be tracked and listed to the User.

.. setting:: CLEANING_WINDOW

``CLEANING_WINDOW``
===================

Default: ``1000``

Amount of Klines and Training Data Time Intervals (observations) to be left for each Symbol after updating indicators. Controls how much data is fed to the Prediction Model. A value of ``1500`` will feed the Prediction Model with at most the last 1500 Training Data and "clean" the older ones.


.. setting:: CLEANING_WINDOW_BOTS_LOGS

``CLEANING_WINDOW_BOTS_LOGS``
=============================

Default: ``3600``

Amount of TraderoBot Logsto be left for each bot after updating  (``0`` to disable logrotate).


.. setting:: PREDICTION_MODEL_CLASS

``PREDICTION_MODEL_CLASS``
==========================

Default: ``base.DecisionTreeRegressor``

Prediction Model Class in dotted path format to be used.


.. setting:: PREDICTION_ENABLED

``PREDICTION_ENABLED``
======================

Default: ``True``

Enables core prediction of the next time interval functionality.

.. setting:: OUTLIERS_MODEL_CLASS

``OUTLIERS_MODEL_CLASS``
========================

Default: ``base.OutliersSVC``

Outliers Model Class in dotted path format to be used.


.. setting:: OUTLIERS_ENABLED

``OUTLIERS_ENABLED``
====================

Default: ``True``

Enables core outliers detection functionality.

.. setting:: OUTLIERS_THRESHOLD

``OUTLIERS_THRESHOLD``
======================

Default: ``0.05``

Proportion of Symbol observations (prices) to be considered as atypical by the outliers detection functionality.


.. setting:: INDICATORS

``INDICATORS``
==============

Default: ``__all__``

Indicators to be enabled (calculated and shown). A string of comma-separated indicators' slugs - i.e. ``macdcg,stp`` - or ``__all__``. For indicator-specific settings, see the indicator's documentation.


.. setting:: BOT_USER_QUOTA

``BOT_USER_QUOTA``
==================

Default: ``0``

Default Bot Quota (Maximum number of Bots) for Users (0 for no quota).


.. setting:: DUMMY_USER_ENABLED

``DUMMY_USER_ENABLED``
======================

Default: ``False``

Enables the Dummy User.


.. setting:: DUMMY_USER_SYMBOL

``DUMMY_USER_SYMBOL``
=====================

Default: ``KEY``

Base Asset of the Symbol to be used at start for Bots.


.. setting:: DUMMY_USER_BOTS

``DUMMY_USER_BOTS``
===================

Default: ``50``

Amount of Bots to create for the Dummy User.


.. setting:: DUMMY_USER_BOT_QUOTA

``DUMMY_USER_BOT_QUOTA``
========================

Default: ``100``

Bot Quota for the Dummy User.
