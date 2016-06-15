`control.py` and `cartpole_sim.py` under this directory are direct translation of the original starter codes.

## Simulator

`cartpole_sim.py` merges what was originally `cart_pole.m`, `get_state.m`, and `show_cart.m` in an OO-manner, and implements two classes:

- `CartPoleSystem` encapsulates the `x`, `x_dot`, `theta`, `theta_dot` tuple into a data class. It is meant to be immutable and the `take_action` method will compute and return a new system state object. The `get_state` method functions just like the original function.
  __Caveat:__  Due to the 1-based nature of MATLAB and 0-based nature of Python, the state number returned by `get_state` has changed to 0-based. Also two possible actions are now 0 and 1 instead of 1 and 2.
- `CartPoleSystem_Render` encapsulates the `show_cart`. As the user just know that calling `render` method will plot a snapshot of the current system state.

## Control Logic

This is the part that you will have to fill in. It is a direct translation of `control.m`. The `plot_learning_curve` function is defined in this file but no need to worry about it. Find places that says "CODE HERE" and fill them in with appropriate code.
