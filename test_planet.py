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
    planet1 = init_planet

    new_mass = 100
    planet1.mass = new_mass

    assert planet1.mass == new_mass

def test_mass_setter_negative_mass(init_planet):
    planet1 = init_planet

    new_mass = -100
    planet1.mass = new_mass

    assert planet1.mass == 0

def test_position_setter(init_planet):
    planet1 = init_planet

    position2 = np.array([1., 0., 0.])
    planet1.position = position2

    assert (planet1.position == position2).all()

def test_position_setter_different_shape(init_planet):
    planet1 = init_planet

    position2 = np.array([1., 0., 0.])
    planet1.position = position2

    assert np.array_equal(planet1.position,position2)
    assert np.array_equiv(planet1.position,position2)
    assert np.allclose(planet1.position,position2)

def test_velocity_setter(init_planet):
    planet1 = init_planet

    velocity2 = np.array([1., 0., 0.])
    planet1.velocity = velocity2

    assert (planet1.velocity == velocity2).all() 

def test_name_setter(init_planet):
    planet1 = init_planet

    name = "padawan"
    planet1.name = name

    assert planet1.name == name
