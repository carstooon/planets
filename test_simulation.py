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