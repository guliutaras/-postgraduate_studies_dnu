import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def backspace():
    print('\r', end='')                     # use '\r' to go back

def draw_progess_bar(n_finished, n_jobs, bar_length=30, sleep_time=0.0):
    finish_percent = int(float((n_finished)) / n_jobs * 100)
    progress_length = int(finish_percent * bar_length /100)
    print ("[%s>%s] %d%%" % ('=' * progress_length, ' ' * (bar_length - progress_length), finish_percent), end='')
    backspace()

    time.sleep(sleep_time)

def samplefuntion2d(n_func, start, finish, rate):
    return n_func(np.linspace(start, finish, rate))

def func_sin(i):
    frequency = 1
    amplitude = 1
    return  amplitude * np.sin(2 * np.pi * frequency * (i))

def func_complex(i):
    return i/(np.sqrt(1+i**2))

def attractorShell(params, t):
    x, y, z = params
    x_dot =-0.4*x+y
    y_dot = x+0.3*y-x*z
    z_dot = -0.2*z+y**2-1
    return x_dot, y_dot, z_dot


def func_ode_sample(start , finish , rate , startx, starty, startz):
    ys = odeint(attractorShell, [startx, starty, startz], np.linspace(start, finish, rate))
    return ys

def plot3Dref(ys):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(ys[:,0], ys[:,1], ys[:,2], alpha=1.75)
