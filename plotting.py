import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Plotting():
    def __init__(self):
        sns.set()
        sns.set_style("white")

    def print_energy(self, df_energy, filename="001_energy.png"):
        f, (ax1, ax2, ax3) = plt.subplots(3)
        f.set_size_inches(8, 11)

        sns.lineplot(x="timestep", y="energy", data=df_energy, ax=ax1)
        sns.lineplot(x="timestep", y="kinetic_energy", data=df_energy, ax=ax2)
        sns.lineplot(x="timestep", y="potential_energy", data=df_energy, ax=ax3)
        ax1.set_ylabel('energy')
        ax2.set_ylabel('kinetic energy')
        ax3.set_ylabel('potential energy')
        ax1.set_xlabel('')
        ax2.set_xlabel('')
        ax3.set_xlabel('t [d]')
        
        f.savefig(filename)

    def print_position(self, df_position, number_of_objects, filename="002_position.png"): 
        fig, ax1 = plt.subplots()
        fig.set_size_inches(8,8)
        
        
        for i in range(number_of_objects):
            sns.scatterplot(x="planet{}_x".format(i), y="planet{}_y".format(i), data=df_position, ax=ax1, palette='green')

        # sns.scatterplot(x="planet0_x", y="planet0_y", data=self.df_position, ax=ax1, palette='green')
        # sns.scatterplot(x="planet1_x", y="planet1_y", hue="timestep", palette="Blues", data=self.df_position, ax=ax1)
        # sns.scatterplot(x="planet2_x", y="planet2_y", hue="timestep", palette="Reds", data=self.df_position, ax=ax1)
        plt.xlabel('x [km]')
        plt.ylabel('y [km]')

        fig.savefig(filename)      