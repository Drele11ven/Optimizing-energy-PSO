import numpy as np

class SimpleBuildingModel:
    def __init__(self, thermal_mass=1.0, heat_loss_coeff=0.5, outdoor_temp_profile=None):
        self.thermal_mass = thermal_mass
        self.heat_loss_coeff = heat_loss_coeff
        if outdoor_temp_profile is None:
            self.outdoor_temp_profile = np.full(24, 15.0)
        else:
            self.outdoor_temp_profile = np.array(outdoor_temp_profile)

    def energy_consumption(self, setpoints, initial_internal_temp=20.0):
        setpoints = np.array(setpoints)
        hours = len(setpoints)
        T = initial_internal_temp
        energy = np.zeros(hours)
        for h in range(hours):
            Tout = self.outdoor_temp_profile[h % len(self.outdoor_temp_profile)]
            desired = setpoints[h]
            delta = desired - T
            hvac_power = max(0.0, delta * self.thermal_mass)   # kW
            loss = self.heat_loss_coeff * (T - Tout)          # kW
            # update internal temp (dt=1h)
            T = T + ( -loss + hvac_power ) / (self.thermal_mass)
            energy[h] = max(0.0, hvac_power)  # انرژی مصرفی گرمایش (kWh) برای آن ساعت
        return energy
