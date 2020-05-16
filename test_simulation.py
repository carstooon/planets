import pytest
import planet
import simulation
import numpy as np

@pytest.fixture
def init_simulation():
    position1 = np.array([0., 0., 0.])
    velocity1 = np.array([0., 0., 0.])
    planet1   = planet.Planet(1E10, position1, velocity1, "Finera")
    
    position2 = np.array([1., 0., 0.])
    velocity2 = np.array([0., 0., 0.])
    planet2   = planet.Planet(1E10, position2, velocity2, "Genobi")

    return simulation.Simulation([planet1, planet2], 100, 0.01)

def test_timesteps_setter(init_simulation):
    sim = init_simulation
    new_timesteps = 1000
    sim.timesteps = new_timesteps

    assert sim.timesteps == new_timesteps

def test_timesteps_setter_not_negative(init_simulation):
    sim = init_simulation
    new_timesteps = -100
    sim.timesteps = new_timesteps

    assert sim.timesteps == 0

def test_timesteps_setter_huge_amount(init_simulation):
    sim = init_simulation
    new_timesteps = 1e100
    sim.timesteps = new_timesteps

    assert sim.timesteps == new_timesteps

def test_propagate_planets1():
    position1 = np.array([0., 0., 0.])
    velocity1 = np.array([1., 0., 0.])
    planet1   = planet.Planet(1E10, position1, velocity1, "Finera")
    
    position2 = np.array([1., 0., 0.])
    velocity2 = np.array([-1., 0., 0.])
    planet2   = planet.Planet(1E10, position2, velocity2, "Genobi")

    position3 = np.array([1., 0., 0.])
    velocity3 = np.array([0., 0., 0.])
    planet3   = planet.Planet(1E10, position3, velocity3, "Kalahan")

    sim = simulation.Simulation([planet1, planet2, planet3], timesteps=1, delta_t=0.01)
    sim.propagate_planets()

    assert np.allclose(planet1.position, np.array([0.01, 0., 0.]))
    assert np.allclose(planet2.position, np.array([1. - 1. * 0.01, 0., 0.]))
    assert np.allclose(planet3.position, np.array([1., 0., 0.]))