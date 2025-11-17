from src.model import SimpleBuildingModel
import numpy as np

def test_energy_nonnegative():
    model = SimpleBuildingModel(thermal_mass=1.0, heat_loss_coeff=0.5, outdoor_temp_profile=np.linspace(5,20,24))
    setpoints = np.full(24, 20.0)
    e = model.energy_consumption(setpoints)
    assert (e >= 0).all()
