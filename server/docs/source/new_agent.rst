Adding a new agent
==================

Non trainable agents
--------------------

-  Create a new controller in ``server/app/agents/controller/``
   inheriting from ``Controller``, implement the step function
-  In controller_manager.py, add the controller to the ``agents_dict``
   dictionary
-  In MARLconfig.json, set the ``agent`` parameter to the name of the
   controller and mode to ``simulation``

Trainable agents
----------------

-  Do the steps above
-  In ``server/app/core/agents/trainables/``, create a new trainable
   agent inheriting from ``Trainable``
-  In ``server/app/services/training_manager.py``, add the trainable to
   the ``agents_dict`` dictionary

Adding configurable parameters
------------------------------

Every controller or agent is given a config_dict parameter in their
``__init__`` and ``act`` methods. To create new propreties for your
agent: 

- Create a new pydantic object in the same .py file as the agent.
- Edit MarlConfig in ``server/app/core/config.py`` to add your new pydantic object as an attribute. - You can export the default config with ``python -m app.core.config`` and copy the output to ``MARLconfig.json`` to see the new parameters.
- In the ``__init__`` method of your agent, add a type hint to the second positional argument (config_dict).
