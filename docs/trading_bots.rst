.. _trading_bots:

============
Trading Bots
============

``tradero`` allows automatizing your trading with **little bots** [1]_.

A bot has fund with a trading strategy, which it will use to decide what action to take - BUY, SELL, JUMP or HOLD - when a decision is requested.

The fund of the bot can be specified in ``Initial Fund (Quote Asset)`` - i.e. 100 BUSD, 15 USDT, XX (:setting:`QUOTE_ASSET`) - and will be re-invested if ``Should Reinvest?`` is checked (and stored in ``Fund (Quote Asset)``).

The bot will trade that fund on a ``Symbol``, storing the amount bought in ``Fund (Baser Asset)``, according to the strategy (``Strategy`` and ``Strategy parameters``) - and will be able to move to others if ``Jumpy?`` is checked.

.. important:: ``DUMMY`` is a mode for the bot where **NO REAL TRANSACTIONS ARE MADE**, instead those transactions are simulated with the current market price of the Symbol and *does not require your BINANCE's API key*. You can test any bot without consecuences using this mode.

You must `generate an API key <https://www.binance.com/en/support/faq/how-to-create-api-keys-on-binance-360002502072>`_ with only ``Reading`` and ``Spot & Margin Trading`` permissions and set it in the User's page in order to have the bot do the trading with `market orders <https://www.binance.com/en/support/faq/what-are-market-order-and-limit-order-and-how-to-place-them-12cba755d6334ad98ced0b66ddde66ec>`_.

.. warning:: **REMEMBER**: You are solely responsible for your trading decisions. You may not press the button for turning ON the bot.

Once all this is set, it can be turned ON.

Every :setting:`TIME_INTERVAL_BOTS` minutes, all the active bots are updated and requested for a decision.

Strategies
==========

A strategy defines when a bot buys, sells or jump [2]_.

ACMadness
---------

Also known as *chicken bots*, this strategy [3]_ relies on the :ref:`stp` indicator (``AC``) for buying.

Given a microgain target, the bot will buy when the predicted accumulated variation for the Symbol is greater than the target and sell when the market price reaches it.

Parameters
^^^^^^^^^^

``microgain``
  The microgain target. Note that this is *gross* microgain, each market order as `a cost <https://www.binance.com/en/fee/trading>`_ - tipically 0.1% without discounts - so a 0.3 target will lead to a net / real microgain of 0.1%

``ac_factor``
  Factor to expand the ``AC``. This is a safety guard, i.e. for a ``ac_factor`` of 2 and a microgain of ``0.3``, the bot will buy if the ``AC`` is greater than 0.6.

``ac_adjusted``
  Whether ponderate (``1``) or not (``0``) the ``AC`` by the Symbol's model score. This is a safety guard, i.e. an ``AC`` of  for a Symbol with a model score of 0.8 will be considered as 0.8.

``ol_prot``
  Outlier protection (``1`` | ``0``). This is safety guard for avoiding buying under unstable conditions - i.e. to prevent a FOMO / price-drop situations.

``max_var_prot``
  Maximum Variation protection. When set greater than 0, it will not buy when the last variation surpasses that threshold (safety guard analogous to Outlier protection)

``keep_going``
  Do not sell if the price has reached the threshold if the ``AC`` is still at a buying point (``1`` | ``0``).


CatchTheWave
-------------

Also known as *dolphin bots*, this strategy [4]_ relies on the :ref:`macd_cg` indicator (``Current Good``) for buying.

Given a Symbol, the bot will buy when it is in "Current Good" and sell when the short-term tendency crosses the middle-term one ("end of the wave").


.. rubric:: References
.. [1] .. autoclass:: base.models.TraderoBot
.. [2] .. autoclass:: base.strategies.TradingStrategy
.. [3] .. autoclass:: base.strategies.ACMadness
.. [4] .. autoclass:: base.strategies.CatchTheWave
