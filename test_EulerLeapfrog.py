import pytest
import planet
import simulation
import solar_system

@pytest.fixture
def init_sun_and_earth():
    solar = solar_system.Solar_System()
    sun_earth          = solar.get_sun_and_earth()
    return sun_earth

def test_sim_sun_and_earth_one_year(init_sun_and_earth):
    number_timesteps = 365*1
    delta_t = 1*24*60*60.

    sim = simulation.EulerLeapfrog(init_sun_and_earth, number_timesteps, delta_t)
    sim.run_simulation()

    begin_earth_x = sim.df.iloc[0]['planet1_x'] # x position of earth in timestep 0
    end_earth_x   = sim.df.iloc[-1]['planet1_x'] # x position of earth in last timestep

    begin_earth_y = sim.df.iloc[0]['planet1_y'] # x position of earth in timestep 0
    end_earth_y   = sim.df.iloc[-1]['planet1_y'] # x position of earth in last timestep

    begin_earth_z = sim.df.iloc[0]['planet1_z'] # x position of earth in timestep 0
    end_earth_z   = sim.df.iloc[-1]['planet1_z'] # x position of earth in last timestep

    assert abs(begin_earth_x - end_earth_x) < 1e7
    assert abs(begin_earth_y - end_earth_y) < 1e7
    assert abs(begin_earth_z - end_earth_z) < 1e7
    