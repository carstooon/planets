import simulation
import plotting as plot
import solar_system

if __name__ == "__main__":
    solar = solar_system.Solar_System()
    sun_earth          = solar.get_sun_and_earth()
    solar_system       = solar.get_solar_system()
    inner_solar_system = solar.get_inner_solar_system()
    outer_solar_system = solar.get_outer_solar_system()

    number_timesteps = 365*1
    delta_t = 1*24*60*60.

    sim = simulation.EulerLeapfrog(sun_earth, number_timesteps, delta_t)
    sim.run_simulation()
    
    print("Start Plotting")
    plot = plot.Plotting()
    plot.print_energy(sim.df)
    plot.print_position(sim.df, len(sim.bodies))
    plot.print_energy_deviation(sim.df)
    # plot.print_EvE(sim.df[["timestep", "planet0_x", "planet0_y", "planet0_z", "planet1_x", "planet1_y", "planet1_z"]])
    # plot.print_EvE(sim.df)