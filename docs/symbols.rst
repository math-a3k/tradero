.. _symbols:

=======
Symbols
=======

A Symbol [1]_ is a market of an asset pair.

An asset pair consist of a base asset and a quote asset.

The base asset will be priced with the quote asset.

I.e. ``ETH/BUSD`` is the market where Etherum is exchanged for Binance USD.

Symbols are loaded or updated with the :ref:`load_symbols` command and are also available through the REST API [2]_ and websockets [3]_.


.. rubric:: References
.. [1] .. autoclass:: base.models.Symbol
.. [2] ``/api/v1/schema/swagger-ui/``
.. [3] ``/ws``
