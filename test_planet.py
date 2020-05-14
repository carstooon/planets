import pytest
import planet
import numpy as np

@pytest.fixture
def init_planet():
    position1 = np.array([0., 0., 0.])
    velocity1 = np.array([0., 0., 0.])
    mass = 1e10
    return planet.Planet(mass, position1, velocity1, "Finera")

def test_mass_setter(init_planet):
    planet1   = init_planet

    new_mass = 100
    planet1.mass = new_mass

    assert(planet1.mass) == new_mass

def test_mass_setter_negative_mass(init_planet):
    planet1   = init_planet

    new_mass = -100
    planet1.mass = new_mass

    assert(planet1.mass) == 0
