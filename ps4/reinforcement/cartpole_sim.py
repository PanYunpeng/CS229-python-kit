from random import random
from math import pi, cos, sin
from textwrap import dedent

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# parameters for simulation dynamics
gravity = 9.8
masscart = 1.0
masspole = 0.3
total_mass = masspole + masscart
length = 0.7  #  actually half the pole's length
polemass_length = masspole * length
force_mag = 10.0
tau = 0.02  #  seconds between state updates
fourthirds = 4 / 3

# Noise parameters
action_flip_prob = 0.00
force_noise_factor = 0.0  # multiplied by between 1-.. and 1+..
no_control_prob = 0.00  # Force is 0 with this probability

# Parameters for state discretization in get_state
one_degree = pi / 180 # 2pi/360
six_degrees = pi / 30
twelve_degrees = pi / 15
fifty_degrees = 5 * pi / 18
total_states = 163


class CartPoleSystem(object):
    def __init__(self, x, x_dot, theta, theta_dot):
        self.x = x
        self.x_dot = x_dot
        self.theta = theta
        self.theta_dot = theta_dot

    def __repr__(self):
        return dedent('''\
        <Cart-Pole system
          x  = {x}
          x' = {x_dot}
          θ  = {theta}
          θ' = {theta_dot}>'''.format(**self.__dict__))

    def take_action(self, action):
        ''' return a new CartPoleSystem object '''
        # Flip action with action_flip_prob
        if random() < action_flip_prob:
            action = 1 - action

        force = force_mag if action > 0 else -force_mag
        force *=  1 - force_noise_factor + 2 * random() * force_noise_factor

        if random() < no_control_prob:
            force = 0

        costheta = cos(self.theta)
        sintheta = sin(self.theta)

        temp = (force + polemass_length *
                (self.theta_dot**2) * sintheta) / total_mass

        thetaacc = (gravity * sintheta - costheta * temp) / (length * (
            fourthirds - masspole * (costheta**2) / total_mass))

        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        # Return the new state variables (using Euler's method)

        new_x = self.x + tau * self.x_dot
        new_x_dot = self.x_dot + tau * xacc
        new_theta = self.theta + tau * self.theta_dot
        new_theta_dot = self.theta_dot + tau * thetaacc

        return CartPoleSystem(new_x, new_x_dot, new_theta, new_theta_dot)

    def get_state(self):
        ''' This function returns a discretized value (a number) for a continuous
        state vector. Currently x is divided into 3 "boxes", x_dot into 3,
        theta into 6 and theta_dot into 3. A finer discretization produces a
        larger state space, but allows a better policy.
        '''
        state = 0

        if self.x < -2.4 or self.x > 2.4 or self.theta < -twelve_degrees or self.theta > twelve_degrees:
            state = total_states - 1
            # to signal failure
        else:
            if self.x < -1.5:
                state = 0
            elif self.x < 1.5:
                state = 1
            else:
                state = 2

            if self.x_dot < -0.5:
                pass
            elif self.x_dot < 0.5:
                state += 3
            else:
                state += 6

            if self.theta < -six_degrees:
                pass
            elif self.theta < -one_degree:
                state += 9
            elif self.theta < 0:
                state += 18
            elif self.theta < one_degree:
                state += 27
            elif self.theta < six_degrees:
                state += 36
            else:
                state += 45

            if self.theta_dot < -fifty_degrees:
                pass
            elif self.theta_dot < fifty_degrees:
                state += 54
            else:
                state += 108

        return state


class CartPoleSystem_Render(object):
    def __init__(self, init_system=None):
        if init_system:
            self._init_render(init_system)

    @staticmethod
    def _compute_pole(cartpole):
        length = 3

        plotx = [
            cartpole.x,
            cartpole.x + length * sin(cartpole.theta)
        ]

        ploty = [
            0,
            length * cos(cartpole.theta)
        ]

        return plotx, ploty

    def _init_render(self, cartpole):
        ''' This function displays the "animation" '''
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.pole, = self.ax.plot(*(self._compute_pole(cartpole)))
        self.cart_top = patches.Rectangle(
            (cartpole.x-0.4, -0.25), # (x,y)
            0.8, 0.25,      # w, h
            facecolor='cyan',
        )
        self.cart_bottom = patches.Rectangle(
            (cartpole.x-0.01, -0.5), # (x,y)
            0.02, 0.25,     # w, h
            facecolor='red',
        )
        self.ax.add_patch(self.cart_top)
        self.ax.add_patch(self.cart_bottom)
        plt.xlim((-3, 3))
        plt.ylim((-0.5, 3.5))

        plt.ion()
        plt.show()

    def render(self, cartpole):
        if not hasattr(self, 'fig'):
            self._init_render(cartpole)
            return

        plotx, ploty = self._compute_pole(cartpole)
        self.pole.set_xdata(plotx)
        self.pole.set_ydata(ploty)

        self.cart_top.set_xy((cartpole.x-0.4, -0.25))
        self.cart_bottom.set_xy((cartpole.x-0.01, -0.5))

        self.fig.canvas.draw()


if __name__ == '__main__':
    c = CartPoleSystem(0, 0, 0, 0)
    r = CartPoleSystem_Render(c)
