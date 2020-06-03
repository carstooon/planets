import numpy as np
import planet

## https://rein.utsc.utoronto.ca/teaching/PSCB57_notes_lecture10.pdf page 58

class Solar_System():
    def __init__(self):        
        position_sun = np.array([ .491403347836458E+05, -3.632715592552171E+05, -1.049148558556447E+04])
        velocity_sun = np.array([6.242516505718979E-3, 1.163260794114081E-2, -2.475374674040771E-4])
        self.sun          = planet.Planet(1.988544E+30, position_sun, velocity_sun, "Sun")

        position_mercure = np.array([ 5.419423385170247E+07, -1.625487416441774E+07, -6.231968220825911E+06])
        velocity_mercure = np.array([4.381057475129131E+00,  4.891120140478765E+01, 3.593097527852676E+00])
        self.planet_mercure   = planet.Planet(3.302E+23, position_mercure, velocity_mercure, "Mercure")

        position_venus = np.array([-2.256798710024889E+07, 1.046561273591759E+08, 2.760223024328955E+06])
        velocity_venus = np.array([-3.431401385797417E+01, -7.710006447409605E+00, 1.875067390915588E+00])
        self.planet_venus   = planet.Planet(4.8685E+24, position_venus, velocity_venus, "Venus")

        position_earth = np.array([-1.418287679667581E+08, 4.126884716814923E+07, -1.125285256157373E+04])
        velocity_earth = np.array([-8.830883522656004E+0, -2.868395352996171E+01, -1.085239827735510E-05])
        self.planet_earth   = planet.Planet(5.97219E+24, position_earth, velocity_earth, "Earth")

        position_mars = np.array([2.408279029085545E+07, 2.311290271817935E+08, 4.261631465386531E+06])
        velocity_mars = np.array([-2.317767379354534E+01,  4.521312752383141E+00, 6.638119814842468E-01])
        self.planet_mars   = planet.Planet( 6.4185E+23, position_mars, velocity_mars, "Mars")

        position_jupiter = np.array([-7.778414202838075E+08,  2.244518699271340E+08,  1.647441732604697E+07])
        velocity_jupiter = np.array([-3.784015648894145E+00, -1.193534649902395E+01, 1.342303598962500E-01])
        self.planet_jupiter   = planet.Planet( 1.89813E+27, position_jupiter, velocity_jupiter, "Jupiter")

        position_saturn = np.array([-2.818805400917871E+08,  1.321479657136385E+09,  -1.177469869993168E+07])
        velocity_saturn = np.array([-9.963111372343775E+00, -2.038111913779875E+00, 4.325059188729626E-01])
        self.planet_saturn   = planet.Planet(  5.68319E+26 , position_saturn, velocity_saturn, "Saturn")

        position_uranus = np.array([ 2.668275057715974E+09,  -1.368207197729832E+09,  -3.965456607834578E+07])
        velocity_uranus = np.array([3.057369253435637E+00,  5.742539618944625E+00, -1.841246675277081E-02])
        self.planet_uranus   = planet.Planet(8.68103E+25, position_uranus, velocity_uranus, "Uranus")

        position_neptune = np.array([  3.068964883542690E+09,  -3.290508769241064E+09,  -2.965492893564129E+06])
        velocity_neptune = np.array([3.938602703572163E+00,  3.739029327757723E+00, -1.671116647134993E-01])
        self.planet_neptune   = planet.Planet( 1.0241E+26, position_neptune, velocity_neptune, "Neptune")

    def get_sun_and_earth(self):
        return [self.sun, self.planet_earth]
    def get_solar_system(self):
        return [self.sun, self.planet_mercure, self.planet_venus, self.planet_earth, self.planet_mars, self.planet_jupiter, self.planet_saturn, self.planet_uranus, self.planet_neptune]    
    def get_inner_solar_system(self):
        return [self.sun, self.planet_mercure, self.planet_venus, self.planet_earth, self.planet_mars]
    def get_outer_solar_system(self):
        return [self.sun, self.planet_jupiter, self.planet_saturn, self.planet_uranus, self.planet_neptune]