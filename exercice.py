#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import math
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



if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    #print(linear_values())
    #print(coordinate_conversion(np.array([(2,3), (5,4)])))
    draw_graphic()
    