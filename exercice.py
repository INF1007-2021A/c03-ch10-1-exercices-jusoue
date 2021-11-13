#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import scipy.integrate as intg
import matplotlib.pyplot as plt


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(start=-1.3, stop=2.5, num=64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    return np.array([(np.sqrt(c[0] ** 2 + c[1] ** 2), np.arctan2(c[1], c[0])) for c in cartesian_coordinates])


def find_closest_index(values: np.ndarray, number: float) -> int:
    return (np.abs(values - number)).argmin()

def draw_graphic():
    x = np.linspace(-1, 1, num=250)
    y = (x ** 2) * np.sin(1 / x ** 2) + x

    plt.scatter(x, y)
    plt.xlim((-1, 1))
    plt.title("I need banana")
    plt.xlabel("x: value of thing")
    plt.ylabel("y: thing but calculated")
    plt.show()

    return 0

def monte_carlo(iteration):
    x_inside_dots = []
    y_inside_dots = []
    x_outside_dots = []
    y_outside_dots = []
    for i in range(iteration):
        x = np.random.random()
        y = np.random.random()
        if np.sqrt(x**2 + y**2) <= 1.0:
            x_inside_dots.append(x)
            y_inside_dots.append(y)
        else:
            x_outside_dots.append(x)
            y_outside_dots.append(y)

    ratio = len(x_inside_dots) / iteration

    plt.scatter(x_inside_dots, y_inside_dots)
    plt.scatter(x_outside_dots, y_outside_dots, c='coral')
    plt.xlim((0, 1))
    plt.title("estimation: " + str(ratio * 4))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    return 0

def integrals():
    evaluation = intg.quad(lambda x : np.exp(-x**2), -np.inf, np.inf)

    x = np.arange(-4, 4, 0.1)
    y = [intg.quad(lambda x : np.exp(-x**2), 0, value)[0] for value in x]

    plt.plot(x, y)
    plt.title(evaluation)
    plt.show()

    return 0

if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    #print(linear_values())
    #print(coordinate_conversion(np.array([(2,3), (5,4)])))
    #draw_graphic()
    #monte_carlo(1000000)
    integrals()