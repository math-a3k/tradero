.. _ai_models:

=======================
Statistical / AI Models
=======================

The main transformation from Klines is the generation of Training Data [1]_, from which the AI models are fed.

Given a ``WINDOW`` - the number of periods to be considered - each Training Data point consists of the percentual variation on open and close (``variation``) and on the high and low (``variation_range``) of the corresponding period and the ``WINDOW`` previous ones.

.. _prediction_model:

Prediction Model
================

A Symbol has a Prediction Model - a Statistical / AI model - associated with it.

The main goal of the Prediction Model is to provide a *Model Score*, this is a technical indicator of how "well" the model can predict or explain the already observed data from previous observations.

This score should be interpreted as an indicator of the "predictable behaviour" of the Symbol in the recent past. "Predictable" in the sense of "consistency", higher model scores mean the Symbol has followed a consistent behaviour - the variation of each period can be "reasonably well" explained by the variation in the previous periods, while lower scores mean "erratic" - the variation of a certain period does not predict the following variation adequately.

It may be also associated with "smoothness", symbols with higher model scores tend to be "smoother" in their evolution.

Comparing the K-lines chart of the Symbol with the highest Model Score to the one with the lowest and one in the middle may help to picture what this indicator refers to.

The Model Score descends when more history is available for training the model. This is likely due to markets behaving differently in different circumstances and more external shocks are taken into account. The amount of history to be used for training can be controlled with the ``CLEANING_WINDOW`` setting, while the  ``--periods`` argument of the :ref:`warm_and_ready` command may be used for providing them start - each period of the command corresponds to one thousand :setting:`TIME_INTERVAL` s, and from recent experience, one is usually good for the current market situation.

Given the abundance of Symbols and the cost of tracking them, a Symbol with a higher Model Score should be preferred to a lower one, as if the conditions of the market do not change - i.e. with an external shock like big bad news or a big troubled whale in the need of cashing out - it is more likely that the market behaviour will be the same in the short term and hence, make a trading decision with more confidence ("A Seguro se lo llevaron preso" [4]_).

Because of this, Symbols are sorted by the model score - internally and in the user interface - and the :setting:`MODEL_SCORE_THRESHOLD` setting controls which symbols are not tracked under that value.

Besides the model score, the Prediction Model provides a prediction for the next time interval (enabled with the :setting:`PREDICTION_ENABLED` setting) and for :ref:`stp`.

Note that there are no confidence intervals provided for the predictions, they are shown for providing a measure of intention (up or down) and its intensity, i.e. an ``AC`` of 10% should indicate that the price is highly likely to go up in the short term.


Implementation details
----------------------

The Training Data transformation allows the use of Regression Models instead of other time series models like SARIMAs.

The Prediction Model should follow the `scikit-learn`_ API and be integrated through `django-ai`_, once this is done, subclass it from ``PredictionModel`` and the corresponding django-ai class [2]_ while setting it in :setting:`PREDICTION_MODEL`.

For the available Regression Models and more information on the default, see `here <https://scikit-learn.org/stable/supervised_learning.html>`_.

.. _outliers:

Outliers Model
==============

An `outlier <https://en.wikipedia.org/wiki/Outlier>`_ is a technical term for "atypical", this means that it is "different" from the "majority" of its class.

If a Symbol had varied between 0.1 and 3 percent in the past intervals, when a variation of 10 percent is seen, that variation is considered to be "atypical" or *outlier*.

Outliers may provide trading opportunities, i.e. a "positive wave" may be starting.

The Outliers Detection Model is used to detect if the last 3 time intervals - considered separately, *O1* is the last one, *O2* the last two, and *O3* the last three - and indicated as blue pills in the upper right corner of the Symbol snippet in the user interface.

The functionality can be enabled with the :setting:`OUTLIERS_ENABLED` and the proportion of observations to be considered as outliers can be controlled with the :setting:`OUTLIERS_THRESHOLD` one, while the model can be swapped with :setting:`OUTLIERS_MODEL`.


Implementation details
----------------------

The Outlier Detection Model should follow the `scikit-learn`_ API and is integrated through `django-ai`_, once this is done, subclass it from ``OutlierDetectionModel`` and the corresponding django-ai class [3]_ while setting it in :setting:`OUTLIERS_MODEL`.

For more comprehensive information on the subject within this context, see `this guide <https://scikit-learn.org/stable/modules/outlier_detection.html>`_.

.. _scikit-learn: https://scikit-learn.org/
.. _django-ai: https://github.com/math-a3k/django-ai/tree/tradero

.. rubric:: References
.. [1] .. autoclass:: base.models.TrainingData
.. [2] .. autoclass:: base.models.DecisionTreeRegressor
.. [3] .. autoclass:: base.models.OutliersSVC
.. [4] "Mr. Safely-Sure was taken to prison"
