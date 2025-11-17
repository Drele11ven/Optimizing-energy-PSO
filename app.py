import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.pso import PSO
from src.model import SimpleBuildingModel
import os
from datetime import datetime
import json
import glob
import csv

# ===========================
# PAGE SETTINGS
# ===========================
st.title("🏠 Building Energy Optimization with PSO")
st.write("Interactive PSO optimization with combined convergence plot for all runs.")

# ===========================
# PSO PARAMETERS
# ===========================
st.sidebar.header("⚙️ PSO Parameters")
num_particles = st.sidebar.slider("Number of Particles", 5, 100, 40)
iterations = st.sidebar.slider("Number of Iterations", 10, 300, 200)
w = st.sidebar.slider("Inertia Weight (w)", 0.1, 1.0, 0.7)
c1 = st.sidebar.slider("Cognitive Coefficient (c1)", 0.1, 3.0, 1.5)
c2 = st.sidebar.slider("Social Coefficient (c2)", 0.1, 3.0, 1.5)
st.sidebar.write("Adjust parameters and click **Run PSO**!")

run_button = st.button("🚀 Run PSO")

# ===========================
# HELPER FUNCTIONS
# ===========================
def tou_price(hour):
    if 17 <= hour <= 21:
        return 0.5
    return 0.2

# ===========================
# RUN PSO
# ===========================
if run_button:
    st.write("⏳ Running optimization...")

    # timestamp و فولدر اختصاصی run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.join("results", f"run_{timestamp}")
    os.makedirs(folder_name, exist_ok=True)

    # مدل ساختمان
    model = SimpleBuildingModel(
        thermal_mass=1.5,
        heat_loss_coeff=0.5,
        outdoor_temp_profile=np.linspace(10, 25, 24)
    )

    dim = 24
    lb = np.full(dim, 18.0)
    ub = np.full(dim, 26.0)

    def objective(x):
        energy = model.energy_consumption(setpoints=x)
        cost = sum(energy[h] * tou_price(h) for h in range(len(energy)))
        comfort_penalty = sum(max(20 - sp, 0) + max(sp - 24, 0) for sp in x)
        return cost + comfort_penalty

    # اجرای PSO
    pso = PSO(n_particles=num_particles, dim=dim, lb=lb, ub=ub,
              max_iter=iterations, w=w, c1=c1, c2=c2)
    best_x, best_val, history = pso.optimize(objective)

    st.success("✔️ Optimization completed successfully!")

    # ذخیره پارامترها و نتایج
    results_dict = {
        "num_particles": num_particles,
        "iterations": iterations,
        "w": w,
        "c1": c1,
        "c2": c2,
        "best_value": best_val,
        "best_setpoints": best_x.tolist(),
        "history": history
    }
    # JSON
    with open(os.path.join(folder_name, "results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
    # CSV
    np.savetxt(os.path.join(folder_name, "best_setpoints.csv"), best_x, delimiter=",")
    np.savetxt(os.path.join(folder_name, "history.csv"), history, delimiter=",")

    st.write(f"✅ All results saved in `{folder_name}`")

    # ===========================
    # نمودار Setpoints
    # ===========================
    fig1, ax1 = plt.subplots()
    ax1.plot(best_x, label="Best Setpoints (°C)")
    ax1.plot(model.outdoor_temp_profile, label="Outdoor Temp (°C)")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Temperature (°C)")
    ax1.grid(True)
    ax1.legend()
    plt.savefig(os.path.join(folder_name, "setpoints_vs_outdoor.png"))
    st.pyplot(fig1)

    # ===========================
    # Convergence نمودار جداگانه
    # ===========================
    fig2, ax2 = plt.subplots()
    ax2.plot(history, label="Current Run")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Best Objective Value")
    ax2.grid(True)
    ax2.set_title("PSO Convergence")
    plt.savefig(os.path.join(folder_name, "pso_convergence.png"))
    st.pyplot(fig2)

    # ===========================
    # Convergence نمودار همه Run ها
    # ===========================
    all_runs = sorted(glob.glob("results/run_*/history.csv"))
    fig_all, ax_all = plt.subplots(figsize=(10,6))
    for run_file in all_runs:
        run_name = os.path.basename(os.path.dirname(run_file))
        h = np.loadtxt(run_file, delimiter=",")
        ax_all.plot(h, label=run_name)
    ax_all.set_xlabel("Iteration")
    ax_all.set_ylabel("Best Objective Value")
    ax_all.set_title("All Runs Convergence")
    ax_all.grid(True)
    ax_all.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results", "convergence_all_runs.png"))
    st.subheader("📊 All Runs Convergence")
    st.pyplot(fig_all)


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
