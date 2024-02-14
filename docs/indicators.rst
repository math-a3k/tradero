.. _indicators:

==========
Indicators
==========

Technical Indicators are calculations done from observed data intended to summarize some aspect of the behaviour of a Symbol in a certain time frame.

Those calculations are used to "understand" the market behaviour, predict its future evolution and make trading decisions.

Once Klines have been received, they are processed to compute the indicators for presentation to the User.

The five core indicators of ``tradero`` are:

  * the :ref:`Model Score <prediction_model>` (MS) (``model_score``),
  * *Variation (last 24h)* (VAR24h) (``last_variation_24h``),
  * *Variation Range (last 24h)* (VR24h) (``variation_range_24h``),
  * :setting:`Quote Asset Volume (24h) <MARKET_SIZE_THRESHOLD>` (VOL24h) (``volume_quote_asset``), and
  * :ref:`Outliers <outliers>`.

Bundled Indicators Plugins
==========================

The indicators plugins to be considered can be controlled with the :setting:`INDICATORS` setting.

.. _macd_cg:

MACD/CG Indicator
-----------------

The MACD/CG is a tradero-developed indicator aimed to detect "positive waves" of Symbols.

It uses the `MACD Indicator <https://en.wikipedia.org/wiki/MACD>`_ with different default parameters (Binance's short, middle, and long-term tendencies calculation) to spot when the short-term is above the middle-term and the middle-term is above the long-term tendency.

Under this situation, it is said that the Symbol is "Current[ly] Good" (**CG**) for a micro-gain trade (the strategy which motivated ``tradero``).

.. image:: macd_cg.png

In the ``tradero`` user interface, this can be spotted when *both* line and signal diff lines are positive (in green).

.. note:: There are slight differences in how Binance and this indicator calculate the tendencies, this may be spotted when the lines are close enough.

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

.. _scg:

SCG Indicator
-------------

The SCG (*Simple Current Good*) is a tradero-developed indicator aimed to detect "positive waves" of Symbols which aims to be the successor of MACD/CG.

It uses simple Moving Averages (MAs) with Binance's short, middle, and long-term tendencies defaults (lines) to spot trading opportunities ("positive waves").

While tracking these tendencies, it calculates their differences, variations, and status: *Current Good* (*CG*) and *Early Onset* (*EO*).

*Current Good* status is when short-term line one is above the middle which is above the long one.

The *SCG index* takes into account the number of periods the Symbol has been in *CG* and the distance of the middle line from the long one. The fewer periods and higher the distance, the higher the value. The index is zero when there is no *CG* status and provides an ordering of preference at the "earliest onset of the positive wave" under the status.

*Early Onset* status is when the short-term line one is only above the middle one (independently of the long-term line).

The *SEO index* (Simple Early Onset) is analogous to the *SCG* but only considers the short and middle-term relation.

.. image:: macd_cg.png

The parameters used for the SCG indicator can be set via the :setting:`SCG` setting.

.. setting:: SCG

``SCG``
^^^^^^^

Default: ``(7, 25, 99)``

The *s*, *m*, *l* parameters for the SCG indicator (middle, long, short) tendencies.


.. _atr:

ATR Indicator
-------------

The ATR (*Average True Range*) is a technical analysis volatility indicator originally developed by J. Welles Wilder, Jr. for commodities. The indicator does not provide an indication of price trend, simply the degree of price volatility. The average true range is an N-period smoothed moving average (SMMA) of the true range values. Wilder recommended a 14-period smoothing [2]_.

The *tradero* implentation calculates the current value as a simple average of the last :setting:`ATR` periods each time interval, it does not update or produces a time series with a weighted average from the previouse value.

The parameters used for the ATR indicator can be set via the :setting:`ATR` setting.

.. setting:: ATR

``ATR``
^^^^^^^

Default: ``14``

The number of periods to be used for the ATR indicator.


.. _dc:

Donchian Channel
----------------

The Donchian channel is an indicator used in market trading developed by Richard Donchian. It is formed by taking the highest high and the lowest low of the last n periods. The area between the high and the low is the channel for the period chosen. [3]_.


The *tradero* implentation also includes the break of the upper and lower bounds for the current time interval.

.. setting:: DC_PERIODS

``DC_PERIODS``
^^^^^^^^^^^^^^

Default: ``20``

The number of periods to be used for the Donchian channel.


.. setting:: DC_AMOUNT

``DC_AMOUNT``
^^^^^^^^^^^^^

Default: ``4``

The amount of time intervals (length of the time serie) to be calculated.


.. _describe:

Describe
--------

The Pandas' DataFrame ``describe`` function plus the current quartile of the symbol's price.


.. setting:: DESCRIBE_PERIODS

``DESCRIBER_PERIODS``
^^^^^^^^^^^^^^^^^^^^^

Default: ``None``

The number of periods to be used for the describe indicator (``None`` for using all the information).



Internal Implementation
=======================

There are two ways of implementing indicators in ``tradero``: into the core or via its plugin architecture [1]_.

.. rubric:: References
.. [1] .. autoclass:: base.indicators.Indicator
.. [2] https://en.wikipedia.org/wiki/Average_true_range
.. [3] https://en.wikipedia.org/wiki/Donchian_channel
