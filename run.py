from src.pso import PSO
from src.model import SimpleBuildingModel
import numpy as np
import matplotlib.pyplot as plt
import os

def tou_price(hour):
    # قیمت فرضی دامی: ساعات اوج (17-21) و بقیه ارزانتر
    if 17 <= hour <= 21:
        return 0.5  # $/kWh اوج
    return 0.2      # $/kWh عادی

def main():
    os.makedirs("results", exist_ok=True)

    model = SimpleBuildingModel(
        thermal_mass=1.5,
        heat_loss_coeff=0.5,
        outdoor_temp_profile = np.linspace(10, 25, 24)
    )

    dim = 24
    lb = np.full(dim, 18.0)
    ub = np.full(dim, 26.0)

    # هزینه شامل مصرف انرژی ضرب در قیمت TOU + جریمه راحتی
    def objective(x):
        energy = model.energy_consumption(setpoints=x)
        # هزینه برق
        cost = sum(energy[h] * tou_price(h) for h in range(len(energy)))
        # جریمه برای خروج از محدوده راحتی 20-24 (مثال): هر درجه خارج 1.0 واحد هزینه اضافه
        comfort_penalty = 0.0
        for h, sp in enumerate(x):
            if sp < 20.0:
                comfort_penalty += (20.0 - sp) * 1.0
            elif sp > 24.0:
                comfort_penalty += (sp - 24.0) * 1.0
        return cost + comfort_penalty

    pso = PSO(n_particles=40, dim=dim, lb=lb, ub=ub, max_iter=200, w=0.7, c1=1.5, c2=1.5)
    best_x, best_val, history = pso.optimize(objective)

    print(f"Best total cost (arb units): {best_val:.3f}")
    # ذخیره نتایج
    np.savetxt("results/best_setpoint.csv", best_x, delimiter=",")
    np.savetxt("results/history.csv", history, delimiter=",")

    # رسم
    hours = np.arange(dim)
    plt.figure()
    plt.plot(hours, best_x, label='best setpoint (°C)')
    plt.plot(hours, model.outdoor_temp_profile, label='outdoor temp (°C)')
    plt.xlabel('Hour')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.title('Optimized Setpoints vs Outdoor Temp')
    plt.grid(True)
    plt.savefig("results/setpoints_vs_outdoor.png", dpi=150)

    plt.figure()
    plt.plot(history)
    plt.xlabel('Iteration')
    plt.ylabel('Best objective so far')
    plt.title('PSO Convergence')
    plt.grid(True)
    plt.savefig("results/convergence.png", dpi=150)

if __name__ == '__main__':
    main()
