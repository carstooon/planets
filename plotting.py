import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Plotting():
    def __init__(self):
        sns.set()
        sns.set_style("white")

    def print_energy(self, df_energy, filename="001_energy.png"):
        f, ax1 = plt.subplots(1)
        sns.lineplot(x="timestep", y="energy", data=df_energy, ax=ax1)
        ax1.set_ylabel('energy')
        ax1.set_xlabel('t [s]')
        f.savefig(filename)

    def print_energy_deviation(self, df, filename="003_energy_deviation.png"):
        f, ax = plt.subplots(1)
        initial_energy = df["energy"].iloc[0]
        df["energy_deviation"] = (df["energy"] - initial_energy) / initial_energy
        scatterplot = sns.lineplot(x="timestep", y="energy_deviation", data=df, ax=ax)
        plt.xlabel('t [s]')
        plt.ylabel('$(E(t) - E_0) / E_0$')
        f.savefig(filename)

    def print_position(self, df, number_of_objects, filename="002_position.png"): 
        fig, ax1 = plt.subplots()
        fig.set_size_inches(8,8)
        
        for i in range(number_of_objects):
            sns.scatterplot(x="planet{}_x".format(i), y="planet{}_y".format(i), data=df, ax=ax1, palette='green')

        plt.xlabel('x [km]')
        plt.ylabel('y [km]')
        fig.savefig(filename)

    def print_EvE(self, df, filename="004_EvE.png"):
        f = sns.pairplot(df)
        plt.savefig(filename)
