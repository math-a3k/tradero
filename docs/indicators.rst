.. _indicators:

==========
Indicators
==========

Technical Indicators are calculations done from observed data which intend to summarize some aspect of the behaviour of a Symbol in a certain time frame.

Those calculations are used to "understand" the market behaviour, predict its future evolution and make trading decisions.

Once Klines have been received, they are processed to compute the indicators for presenting to the User.

The six main indicators of ``tradero`` are:

  * the :ref:`Model Score <prediction_model>` (MS) (``model_score``),
  * *Variation (last 24h)* (VAR24h) (``last_variation_24h``),
  * :setting:`Quote Asset Volume (24h) <MARKET_SIZE_THRESHOLD>` (VOL24h) (``volume_quote_asset``),
  * :ref:`Outliers <outliers>`, and the
  * :ref:`macd_cg`.

Bundled Indicators Plugins
==========================

The indicators plugins to be considered can be controlled with the :setting:`INDICATORS` setting.

.. _macd_cg:

MACD/CG Indicator
-----------------

The MACD/CG is tradero-developed indicator aimed to detect "positive waves" of Symbols.

It uses the `MACD Indicator <https://en.wikipedia.org/wiki/MACD>`_ with different default parameters (Binance's short, middle and long term tendencies calculation) to spot when the short term is above the middle term and the middle term is above the long term tendency.

Under this situation, it is said that the Symbol is "Current[ly] Good" (**CG**) for a micro-gain trade (the strategy which motivated ``tradero``).

.. image:: macd_cg.png

In the ``tradero`` user interface this can be spoted when *both* line and signal diff lines are positive (in green).

.. note:: There are slight differences on how Binance and tradero calculate the tendencies, this may be spotted when the lines are close enough.

The parameters used for the MACD indicator can be set via the :setting:`MACD_CG` setting.

.. setting:: MACD_CG

``MACD_CG``
^^^^^^^^^^^

Default: ``(25, 99, 7)``

The *a*, *b*, *c* parameters for the MACD indicator (middle, long, short) tendencies.

.. _stp:

STP (Short Term Prediction)
---------------------------

This indicator uses the :ref:`prediction_model` (which has to be :setting:`enabled <PREDICTION_ENABLED>`) to predict the next :setting:`STP_PERIODS`.

It aims to provide an indicator of the intensity of the current tendency of the Symbol.

The **AC** value corresponds to the accumulated of the predicted variation in those periods.

.. note:: An ``AC`` of 10% should be read as "the price is highly likely to go up in the short term", not "the price will increase 10% in the next 4 periods".

.. setting:: STP_PERIODS

``STP_PERIODS``
^^^^^^^^^^^^^^^

Default: ``4``

The amount of time intervals to be predicted.


Internal Implementation
=======================

There are two ways of implementing indicators in ``tradero``: into the core or via its plugin architecture [1]_.

.. rubric:: References
.. [1] .. autoclass:: base.indicators.Indicator
