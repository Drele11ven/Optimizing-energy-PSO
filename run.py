import numpy as np
import matplotlib.pyplot as plt
from src.pso import PSO
from src.model import SimpleBuildingModel
import os
from datetime import datetime
import json
import time
import csv
import glob

# ===========================
# مدل ساختمان
# ===========================
model = SimpleBuildingModel(
    thermal_mass=1.5,
    heat_loss_coeff=0.5,
    outdoor_temp_profile=np.linspace(10, 25, 24)
)

def tou_price(hour):
    return 0.5 if 17 <= hour <= 21 else 0.2

def objective(x):
    energy = model.energy_consumption(setpoints=x)
    cost = sum(energy[h] * tou_price(h) for h in range(len(energy)))
    comfort_penalty = sum(max(20 - sp, 0) + max(sp - 24, 0) for sp in x)
    return cost + comfort_penalty

# ===========================
# پارامترهای 15 run
# ===========================
param_list = [
    {"num_particles": 20, "iterations": 50, "w": 0.7, "c1": 1.5, "c2": 1.5},
    {"num_particles": 25, "iterations": 60, "w": 0.8, "c1": 1.8, "c2": 1.8},
    {"num_particles": 30, "iterations": 70, "w": 0.6, "c1": 1.4, "c2": 1.6},
    {"num_particles": 35, "iterations": 80, "w": 0.7, "c1": 1.6, "c2": 1.7},
    {"num_particles": 40, "iterations": 200, "w": 0.7, "c1": 1.5, "c2": 1.5},
    {"num_particles": 40, "iterations": 90, "w": 0.75, "c1": 1.5, "c2": 1.8},
    {"num_particles": 45, "iterations": 100, "w": 0.7, "c1": 1.4, "c2": 1.6},
    {"num_particles": 50, "iterations": 110, "w": 0.65, "c1": 1.7, "c2": 1.5},
    {"num_particles": 55, "iterations": 120, "w": 0.7, "c1": 1.5, "c2": 1.5},
    {"num_particles": 60, "iterations": 130, "w": 0.75, "c1": 1.6, "c2": 1.4},
    {"num_particles": 65, "iterations": 140, "w": 0.8, "c1": 1.8, "c2": 1.6},
    {"num_particles": 70, "iterations": 150, "w": 0.7, "c1": 1.5, "c2": 1.7},
    {"num_particles": 75, "iterations": 160, "w": 0.65, "c1": 1.6, "c2": 1.6},
    {"num_particles": 80, "iterations": 170, "w": 0.7, "c1": 1.5, "c2": 1.5},
    {"num_particles": 85, "iterations": 180, "w": 0.75, "c1": 1.7, "c2": 1.8},
    {"num_particles": 90, "iterations": 200, "w": 0.7, "c1": 1.6, "c2": 1.5}
]

# ===========================
# اجرای batch با لیبل فولدر
# ===========================
all_histories = []
all_labels = []

for i, params in enumerate(param_list, 1):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.join("results", f"run_{timestamp}")
    os.makedirs(folder_name, exist_ok=True)

    dim = 24
    lb = np.full(dim, 18.0)
    ub = np.full(dim, 26.0)

    pso = PSO(
        n_particles=params["num_particles"],
        dim=dim,
        lb=lb,
        ub=ub,
        max_iter=params["iterations"],
        w=params["w"],
        c1=params["c1"],
        c2=params["c2"]
    )

    best_x, best_val, history = pso.optimize(objective)

    # ذخیره نتایج JSON
    results_dict = {
        "num_particles": params["num_particles"],
        "iterations": params["iterations"],
        "w": params["w"],
        "c1": params["c1"],
        "c2": params["c2"],
        "best_value": best_val,
        "best_setpoints": best_x.tolist(),
        "history": history
    }
    with open(os.path.join(folder_name, "results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)

    # ذخیره CSV
    np.savetxt(os.path.join(folder_name, "best_setpoints.csv"), best_x, delimiter=",")
    np.savetxt(os.path.join(folder_name, "history.csv"), history, delimiter=",")

    all_histories.append(history)
    # استفاده از اسم فولدر به جای شماره run
    all_labels.append(os.path.basename(folder_name))

    time.sleep(1.1)

# ===========================
# رسم نمودار کلی Convergence
# ===========================
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(12,6))
for history, label in zip(all_histories, all_labels):
    plt.plot(history, label=label)
plt.xlabel("Iteration")
plt.ylabel("Best Objective Value")
plt.title("PSO Convergence for All Runs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("results", "convergence_all_runs.png"))
plt.show()


all_json_files = sorted(glob.glob("results/run_*/results.json"))

csv_file = os.path.join("results", "all_results.csv")
with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    # هدر
    writer.writerow(["folder_name", "num_particles", "iterations", "w", "c1", "c2", "best_value"])

    for json_file in all_json_files:
        with open(json_file, "r") as jf:
            data = json.load(jf)
            folder_name = os.path.basename(os.path.dirname(json_file))
            writer.writerow([
                folder_name,
                data["num_particles"],
                data["iterations"],
                data["w"],
                data["c1"],
                data["c2"],
                data["best_value"]
            ])

print(f"✅ All JSON results merged into {csv_file}")
