import pytest
import planet
import simulation
import numpy as np
import math

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

# def test_run_simulation(init_simulation):
#     sim = init_simulation
#     sim.run_simulation()

#     assert np.allclose(sim.planets[0].velocity, -1 * sim.planets[1].velocity)
    
#     assert np.allclose(sim.planets[0].position, np.array([0.33037785, 0., 0.]))
#     assert np.allclose(sim.planets[1].position, np.array([0.66962215, 0., 0.]))



@pytest.fixture
def init_RungeKutta4():
    position1 = np.array([0., 0., 0.])
    velocity1 = np.array([0., 0., 0.])
    planet1   = planet.Planet(1E10, position1, velocity1, "Finera")
    
    position2 = np.array([1., 0., 0.])
    velocity2 = np.array([0., 0., 0.])
    planet2   = planet.Planet(1E10, position2, velocity2, "Genobi")

    position3 = np.array([1., 0., 0.])
    velocity3 = np.array([0., 0., 0.])
    planet3   = planet.Planet(1E10, position3, velocity3, "Endor")

    return simulation.RungeKutta4([planet1, planet2], 0.01, 0.001)

def test_precision_setter(init_RungeKutta4):
    sim = init_RungeKutta4
    new_precision = 0.0001
    sim.precision = new_precision

    assert sim.precision == new_precision

def test_precision_setter_not_negative(init_RungeKutta4):
    sim = init_RungeKutta4
    new_precision = -100
    sim.precision = new_precision

    assert sim.precision == 0.01

def test_fInternal_x(init_RungeKutta4):
    sim = init_RungeKutta4
    
    assert sim.fInternal(0,1,0,sim.planets) > 0



def test_fInternal_y(init_RungeKutta4):
    sim = init_RungeKutta4
    
    assert sim.fInternal(0,1,1,sim.planets) == 0


def test_fInternal3D(init_RungeKutta4):
    sim = init_RungeKutta4
    f_int = sim.fInternal3D(0,1,sim.planets)

    assert f_int[0] == sim.fInternal(0,1,0,sim.planets)
    assert f_int[1] == sim.fInternal(0,1,1,sim.planets)
    assert f_int[2] == sim.fInternal(0,1,2,sim.planets)

def test_fInternal_exception(init_RungeKutta4):
    sim = init_RungeKutta4
    
    assert sim.fInternal(0,0,1,sim.planets) == 0

def test_fExternal(init_RungeKutta4):
    sim = init_RungeKutta4
    
    assert sim.fExternal(0,1,sim.planets) == 0


def test_Energy(init_RungeKutta4):
    sim = init_RungeKutta4
    
    assert sim.Energy() < 0

def test_CalcAcceleration(init_RungeKutta4):
    sim = init_RungeKutta4
    
    assert sim.CalcAcceleration(0,0,sim.planets) > 0

def test_CalcAcceleration3D(init_RungeKutta4):
    sim = init_RungeKutta4
    
    acc = sim.CalcAcceleration3D(0,sim.planets)
    assert acc[0] == sim.CalcAcceleration(0,0,sim.planets) 


def test_NextStep_Planet0(init_RungeKutta4):
    sim = init_RungeKutta4

    pos_x = sim.planets[0].position[0]
    pos_y = sim.planets[0].position[1]
    pos_z = sim.planets[0].position[2]

    vel_x = sim.planets[0].velocity[0]
    vel_y = sim.planets[0].velocity[1]
    vel_z = sim.planets[0].velocity[2]

    sim.NextStep()

    assert sim.planets[0].position[0] != pos_x
    assert sim.planets[0].velocity[0] != vel_x
    assert sim.planets[0].position[1] == pos_y
    assert sim.planets[0].velocity[1] == vel_y

def test_NextStep_Planet1(init_RungeKutta4):
    sim = init_RungeKutta4

    pos_x = sim.planets[1].position[0]
    pos_y = sim.planets[1].position[1]
    pos_z = sim.planets[1].position[2]

    vel_x = sim.planets[1].velocity[0]
    vel_y = sim.planets[1].velocity[1]
    vel_z = sim.planets[1].velocity[2]

    sim.NextStep()

    assert sim.planets[1].position[0] != pos_x
    assert sim.planets[1].velocity[0] != vel_x
    assert sim.planets[1].position[1] == pos_y
    assert sim.planets[1].velocity[1] == vel_y

def test_NextStep3D(init_RungeKutta4):
    sim = init_RungeKutta4
    sim.NextStep()

    pos_x = sim.planets[1].position[0]
    pos_y = sim.planets[1].position[1]
    pos_z = sim.planets[1].position[2]

    vel_x = sim.planets[1].velocity[0]
    vel_y = sim.planets[1].velocity[1]
    vel_z = sim.planets[1].velocity[2]

    sim2 = init_RungeKutta4
    sim2.NextStep3D()

    pos_x2 = sim2.planets[1].position[0]
    pos_y2 = sim2.planets[1].position[1]
    pos_z2 = sim2.planets[1].position[2]

    vel_x2 = sim2.planets[1].velocity[0]
    vel_y2 = sim2.planets[1].velocity[1]
    vel_z2 = sim2.planets[1].velocity[2]

    assert math.isclose(pos_x,pos_x2)
    assert math.isclose(pos_y,pos_y2)
    assert math.isclose(pos_z,pos_z2)

    #assert math.isclose(vel_x,vel_x2)
    #assert math.isclose(vel_y,vel_y2)
    #assert math.isclose(vel_z,vel_z2)

    

def test_NextStepError1(init_RungeKutta4):
    sim = init_RungeKutta4
    delta_t   = sim.delta_t
    time = sim.simulated_time

    sim.NextStepError1()

    assert sim.delta_t != delta_t
    assert sim.simulated_time != time

def test_NextStepError2(init_RungeKutta4):
    sim = init_RungeKutta4

    delta_t   = sim.delta_t
    time = sim.simulated_time

    sim.NextStepError2()

    print(sim.delta_t)
    print(time)
    assert sim.delta_t != delta_t
    assert sim.simulated_time != time

def test_RunSimulation(init_RungeKutta4):
    sim = init_RungeKutta4

    run_time = 100
    sim.RunSimulation(run_time)

    assert sim.simulated_time >= run_time
