import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Plotting():
    def __init__(self):
        sns.set()
        sns.set_style("white")

    def print_energy(self, df_energy, filename="001_energy.png"):
        f, (ax1, ax2, ax3) = plt.subplots(3)
        f.set_size_inches(8, 10)

        sns.lineplot(x="timestep", y="energy", data=df_energy, ax=ax1)
        sns.lineplot(x="timestep", y="kinetic_energy", data=df_energy, ax=ax2)
        sns.lineplot(x="timestep", y="potential_energy", data=df_energy, ax=ax3)
        ax1.set_ylabel('energy')
        ax2.set_ylabel('kinetic energy')
        ax3.set_ylabel('potential energy')
        ax1.set_xlabel('')
        ax2.set_xlabel('')
        ax3.set_xlabel('t [timesteps]')
        f.savefig(filename)

    def print_energy_deviation(self, df_energy, filename="003_energy_deviation.png"):
        f, ax = plt.subplots(1)
        initial_energy = df_energy["energy"].iloc[0]
        df_energy["energy_deviation"] = (df_energy["energy"] - initial_energy) / initial_energy
        scatterplot = sns.lineplot(x="timestep", y="energy_deviation", data=df_energy, ax=ax)
        plt.xlabel('t [timesteps]')
        plt.ylabel('$(E(t) - E_0) / E_0$')
        f.savefig(filename)

    def print_position(self, df_position, number_of_objects, filename="002_position.png"): 
        fig, ax1 = plt.subplots()
        fig.set_size_inches(8,8)
        
        for i in range(number_of_objects):
            sns.scatterplot(x="planet{}_x".format(i), y="planet{}_y".format(i), data=df_position, ax=ax1, palette='green')

        plt.xlabel('x [km]')
        plt.ylabel('y [km]')
        fig.savefig(filename)      