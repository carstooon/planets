import simulation
import plotting as plot
import solar_system

if __name__ == "__main__":
    solar = solar_system.Solar_System()
    sun_earth          = solar.get_sun_and_earth()
    solar_system       = solar.get_solar_system()
    inner_solar_system = solar.get_inner_solar_system()
    outer_solar_system = solar.get_outer_solar_system()

    number_timesteps = 365*24
    delta_t = 1*60*60.

    sim = simulation.Simulation(sun_earth, number_timesteps, delta_t)
    sim.run_simulation()
    
    plot = plot.Plotting()
    plot.print_energy(sim.df_energy)
    plot.print_position(sim.df_position, len(sim.planets))
    plot.print_energy_deviation(sim.df_energy)
