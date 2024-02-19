import math

"""Simulations.py

Acoustic Simulation Utilities

Author : Logan R Boehm
"""

def geometric_divergence(distance: float, d_0: float = 1) -> float:
    """
    Given a distance, calculate the decibel loss over that distance,
    considering a spherical wavefront

    distance : Distance (meters)
    d_0 : Reference distance (meters)

    Returns dB
    """
    return 20 * math.log10(distance / d_0) + 11


if __name__ == "__main__":
    rng = 10
    print(geometric_divergence(rng))