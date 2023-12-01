.. _trading_bots:

============
Trading Bots
============

``tradero`` allows automatizing your trading with **little bots** [1]_.

A bot has a fund with a trading strategy, which it will use to decide what action to take - BUY, SELL, JUMP, or HOLD - when a decision is requested.

The fund of the bot can be specified in ``Initial Fund (Quote Asset)`` - i.e. 100 BUSD, 15 USDT, XX (:setting:`QUOTE_ASSET`) - and will be re-invested if ``Should Reinvest?`` is checked (and stored in ``Fund (Quote Asset)``).

The bot will trade that fund on a ``Symbol``, storing the amount bought in ``Fund (Baser Asset)``, according to the strategy (``Strategy`` and ``Strategy parameters``) - and will be able to move to others if ``Jumpy?`` is checked.

.. important:: ``DUMMY`` is a mode for the bot where **NO REAL TRANSACTIONS ARE MADE**, instead, those transactions are simulated with the current market price of the Symbol and *do not require your BINANCE's API key*. You can test any bot without consequences using this mode.

You must `generate an API key <https://www.binance.com/en/support/faq/how-to-create-api-keys-on-binance-360002502072>`_ with only ``Reading`` and ``Spot & Margin Trading`` permissions and set it in the User's page to have the bot do the trading with `market orders <https://www.binance.com/en/support/faq/what-are-market-order-and-limit-order-and-how-to-place-them-12cba755d6334ad98ced0b66ddde66ec>`_.

.. warning:: **REMEMBER**: You are solely responsible for your trading decisions. You may not press the button to turn ON the bot.

Once all this is set, it can be turned ON.

Every :setting:`TIME_INTERVAL_BOTS` minutes, all the active bots are updated and requested for a decision.

Strategies
==========

A strategy defines when a bot buys, sells, or jumps [2]_.

ACMadness
---------

Also known as *chicken bots*, this strategy [3]_ relies on the :ref:`stp` indicator (``AC``) for buying.

Given a micro gain target, the bot will buy when the predicted accumulated variation for the Symbol is greater than the target and sell when the market price reaches it.

Parameters
^^^^^^^^^^

``microgain``
  The micro gain target. Note that this is a *gross* micro gain, each market order has `a cost <https://www.binance.com/en/fee/trading>`_ - typically 0.1% without discounts - so a 0.3 target will lead to a net/real micro gain of 0.1%

``ac_factor``
  Factor to expand the ``AC``. This is a safety guard, i.e. for an ``ac_factor`` of 2 and a ``microgain`` of ``0.3``, the bot will buy if the ``AC`` is greater than 0.6.

``ac_adjusted``
  Whether ponders (``1``) or not (``0``) the ``AC`` by the Symbol's model score. This is a safety guard, i.e. an ``AC`` of  for a Symbol with a model score of 0.8 will be considered as 0.8.

``ol_prot``
  Outlier protection (``1`` | ``0``). This is a safety guard for avoiding buying under unstable conditions - i.e. to prevent a FOMO / price-drop situation.

``max_var_prot``
  Maximum Variation protection. When set greater than 0, it will not buy when the last variation surpasses that threshold (safety guard analogous to Outlier protection)

``keep_going``
  Do not sell if the price has reached the threshold if the ``AC`` is still at a buying point (``1`` | ``0``).

``vr24h_max``
  Maximum Variation Range (24h) of a Symbol to buy (``Decimal``, defaults to ``10``) if enabled (greater than zero). Meant to keep the bot out of the market when it is not moving sideways.


CatchTheWave
-------------

Also known as *dolphin bots*, this strategy [4]_ relies on the :ref:`scg` indicator for buying.

Given a Symbol, the bot will buy when it is on "good status" while selling when the short-term line crosses the middle-term one ("end of the wave").

*Good Status* is either when the Symbol is at "Current Good" and the short-term tendency is increasing ("wave onset") or at "Early Onset" and the long-term line is not decreasing.

This strategy uses the bots' local memory to track the price movement of the asset at its native time resolution (:setting:`TIME_INTERVAL_BOTS`) while using the indicators from the matrix (also known as *Rainha Botzinha*) at her time resolution (:setting:`TIME_INTERVAL`) to foresee the market situation.

Parameters
^^^^^^^^^^

``early_onset``
  Use the *Early Onset* status (``1`` | ``0``, defaults to ``0``).

``sell_on_maxima``
  Sell after the short-term tendency has local maxima (``1`` | ``0``, defaults to ``1``).

``onset_periods``
  Periods to consider the wave is on onset (short-term increasing) (``int``, defaults to ``2``).

``maxima_tol``
  Tolerance (in percentage) for the short-term line variation to consider it as local maxima (``Decimal``, defaults to ``0.1``).

``sell_safeguard``
  The extra percentage of the buying price to set as the min. selling threshold for automatic selling in the worst-case scenario (``Decimal``, defaults to ``0.3``).

``use_local_memory``
  Use the bot's local memory (``1`` | ``0``, defaults to ``1``)

``use_matrix_time_res``
  Use matrix's time resolution (:setting:`TIME_INTERVAL`) (``1`` | ``0``, defaults to ``0``)

``vr24h_min``
  Minimum Variation Range (24h) of a Symbol to buy (``Decimal``, defaults to ``3``) if enabled (greater than zero). Meant to keep the bot out of the market when it is moving sideways narrowly.


Turtle
------

Also known as *little turtle bots*, this strategy [5]_ relies on the :ref:`scg`, ref:`atr`, and ref:`dc` indicators for deciding.

Inspired by this `video <https://www.youtube.com/watch?v=X9edzFqmUyk>`_, this is the implementation of the strategy proposed by Richard Dennis and William Eckhardt in the 1980s, using the `Donchian channel <https://admiralmarkets.com/es/education/articles/forex-indicators/lo-que-todos-deberian-saber-sobre-el-indicador-de-canal-donchian>`_.

Given a Symbol, the bot will buy when it is on "good status" (the middle-term tendency is ascending) and an upper break of the Donchian channel has occurred, while selling when a lower break of the channel occurs or a stop-loss is executed when the price reaches the buying price minus 2 ATRs.

Parameters
^^^^^^^^^^

``use_matrix_time_res``
  Use matrix's time resolution (:setting:`TIME_INTERVAL`) (``1`` | ``0``, defaults to ``0``)

``vr24h_min``
  Minimum Variation Range (24h) of a Symbol to buy (``Decimal``, defaults to ``3``) if enabled (greater than zero). Meant to keep the bot out of the market when it is moving sideways narrowly.


.. rubric:: References
.. [1] .. autoclass:: base.models.TraderoBot
.. [2] .. autoclass:: base.strategies.TradingStrategy
.. [3] .. autoclass:: base.strategies.ACMadness
.. [4] .. autoclass:: base.strategies.CatchTheWave
.. [5] .. autoclass:: base.strategies.Turtle
