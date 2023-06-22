.. _kllnes:

=============
Klines (Data)
=============

*K-line*, also known as candlestick, is the unit representation of a chart that plots the opening price, closing price, highest price, and lowest price on each period to reflect price evolution. It is said that the K-line was invented by a Japanese rice trader called Honma [1]_.

Data of price (and others) in a certain time frame of a Symbol is retrieved and stored from the `Exchange`_ in the form of K-line [2]_.

.. _Exchange: https://www.binance.com/

.. rubric:: References
.. [1] https://support.coinex.com/hc/en-us/articles/8747125554329-Introduction-to-K-line
.. [2] .. autoclass:: base.models.Kline
