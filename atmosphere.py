import math

"""Atmosphere.py

Atmospheric utilities

Using documentation from:
ISO-9613 Version:1993 (There are 2024 versions, $200 to read them though)

Buck, A. L., 1981: New Equations for Computing Vapor Pressure and Enhancement Factor. J. Appl. Meteor. Climatol., 20, 1527–1532, https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2.


Author : Logan R Boehm
"""

# Reference International Standard ISO-9613 Part 1 for related documentation

GAS_CONSTANT = 8.314 # L kPa / mol K

REF_AMBIENT_PRESS = 101.325 # kPa
REF_AMBIENT_TEMP = 293.15 # K

def celsius_to_kelvin(temp_c: float) -> float:
    """
    Convert a temperature in Celsius to Kelvin
    """
    return temp_c + 273.15

def kelvin_to_celsius(temp_k: float) -> float:
    """
    Convert a temperature in Kelvin to Celsius
    """
    return temp_k - 273.15

def relative_to_molar_conc_humidity(temperature : float, pressure: float, rel_hum: float) -> float:
    """
    Given a temperature, pressure and relative humidity,
    calculate the molar concentration of water in the
    air as a percentage.

    temperature : Temperature (kelvin)
    pressure : Pressure (kiloPascals)
    rel_hum : Relative humidity (0 to 1)

    Returns range from 0% to 100%
    """

    # Use the Arden Buck equation to calculate the saturation vapor pressure
    t_c = kelvin_to_celsius(temperature)
    if t_c < 0:
        # Temperature is below freezing, convention dictates
        # we evaluate relative humidity over ice, not water
        vapor_pressure = .61115 * math.exp((23.036 - t_c / 333.7) * (t_c / (279.82 + t_c)))
    else:
        vapor_pressure = .61121 * math.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))

    # ISO 9613 Part 1, Annex B Formula 1
    molar_concentration = rel_hum * (vapor_pressure / REF_AMBIENT_PRESS) / (pressure / REF_AMBIENT_PRESS)

    return molar_concentration * 100

# NOTE : Relaxation frequencies correspond to how certain molecules interact with sound waves
# Some of the waves energy causes certain molecules to vibrate and rotate at their relaxation frequency
# Thus some energy is absorbed from the wave. We need to account for this.

# We therefore calculate the relaxation frequency for N2 and O2, since they make up around 99% of the atmosphere

def oxygen_relaxation_frequency(pressure : float, water_molar_conc : float):
    """
    Calculate the relaxation frequency of the O2 molecule using:

    pressure : Pressure (kPa)
    water_molar_conc : Molar concentration of water (0 to 1)
    """
    # ISO-9613 Part 1 Equation 3
    return  (pressure / REF_AMBIENT_PRESS) * (24 + (4.04 * math.pow(10, 4) * water_molar_conc * (.02 + water_molar_conc) / (.391 + water_molar_conc)))

def nitrogen_relaxation_frequency(pressure : float, temperature : float, water_molar_conc : float):
    """
    Calculate the relaxation frequency of the N2 molecule using:

    pressure : Pressure (kPa)
    temperature : Temperature (Kelvin)
    water_molar_conc : Molar concentration of water (0 to 1)
    """
    # ISO-9613 Part 1 Equation 4
    p_1 = (pressure / REF_AMBIENT_PRESS) / math.sqrt(temperature / REF_AMBIENT_TEMP)
    return p_1 * (9 + 280 * water_molar_conc * math.exp(-4.170 * (math.pow(temperature / REF_AMBIENT_TEMP, -1 / 3) - 1)))

class Atmosphere:
    """
    Data class for atmospheric related constants.
    """
    def __init__(self, temp_c : float, press_atm : float, rel_humidity : float = 0):
        """
        Create an atmosphere reference with:

        temp_c : temperature (celsius)
        press_atm : pressure (atmospheres)
        rel_humidity : relative humidity (0 to 1)
        """
        self.temp_k = celsius_to_kelvin(temp_c)
        self.press_kpa = press_atm * 101.325
        self.h20_molar_conc = relative_to_molar_conc_humidity(self.temp_k, self.press_kpa, rel_humidity)
        self.oxy_relax = oxygen_relaxation_frequency(self.press_kpa, self.h20_molar_conc)
        self.nitro_relax = nitrogen_relaxation_frequency(self.press_kpa, self.temp_k, self.h20_molar_conc)
        self.speed_of_sound = 343.2 * math.sqrt(self.temp_k / REF_AMBIENT_TEMP)

    @staticmethod
    def stp():
        """
        0° Celsius, 1 atm pressure, no humidity
        """
        return Atmosphere(0, 1, 0)

    @staticmethod
    def from_temperature(self, temp_c : float, rel_humidity : float = 0):
        """
        Assuming standard pressure of 1 atm, create atmosphere

        temp_c : temperature (celsius)
        rel_humidity : relative humidity (0 to 1)
        """
        return Atmosphere(temp_c, 1, rel_humidity = rel_humidity)

    def attenuation_coefficient(self, frequency : float):
        """
        Given the frequency of an acoustic wave in Hertz, calculate the
        atmospheric absorption in dB / meter
        """
        # ISO-9613 Part 1 Equation 5
        f_squared = math.pow(frequency, 2)
        oxy_comp = .01275 * math.exp(-2239.1 / self.temp_k) / (self.oxy_relax + (f_squared / self.oxy_relax))
        nitro_comp = .1068 * math.exp(-3352 / self.temp_k) / (self.nitro_relax + (f_squared / self.nitro_relax))
        relax = math.pow(self.temp_k / REF_AMBIENT_TEMP, -5 / 2) * (oxy_comp + nitro_comp)
        atmo = 1.84 * math.pow(10, -11) * math.sqrt(self.temp_k / REF_AMBIENT_TEMP) / (self.press_kpa / REF_AMBIENT_PRESS)
        return 8.686 * f_squared * (atmo + relax)

if __name__ == "__main__":
    atmo = Atmosphere(15, 1, .2)
    k = -2
    tgt_midband = 1000 * math.pow(math.pow(10, 3 * (1/3) / 10), k)
    print(tgt_midband)
    atten = 1000 * atmo.attenuation_coefficient(tgt_midband)
    print(atten)
    print(3.8)
