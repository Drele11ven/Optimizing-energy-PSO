# Optimizing Energy Consumption with PSO

## 1️⃣ Project Introduction
This project optimizes the energy consumption of a simple building using the Particle Swarm Optimization (PSO) algorithm and displays results via an interactive Streamlit interface. The user can change algorithm parameters and observe their effect on the best temperature setpoints and PSO convergence.

---

## 2️⃣ Activate the Environment

Windows:
```bash
venv\Scripts\activate
```

macOS / Linux:
```bash
source venv/bin/activate
```

---

## 3️⃣ Install Packages
```bash
pip install -r requirements.txt
```

> If you don't have `requirements.txt`, you can install required packages:
```bash
pip install numpy matplotlib streamlit
```

---

## 🚀 Run the Project

To run the Streamlit app:
```bash
streamlit run app.py
```
> `app.py` is your main Streamlit file.

---

## 📁 Project Structure
```
Optimizing-energy-PSO/
│
├─ src/
│   ├─ model.py        # building model and energy consumption calculation
│   └─ pso.py          # PSO algorithm implementation
│
├─ results/            # store JSON/CSV results and plots
│
├─ app.py              # Streamlit interface
├─ requirements.txt    # list of required packages
└─ README.md
```

---

## 📊 Outputs

- Each run of the algorithm is saved to `results/run_TIMESTAMP/`.
- Output files include:
    - `results.json`: PSO parameters and convergence history
    - `best_setpoints.csv`: best temperature setpoints for each hour
    - `history.csv`: history of the best objective value per iteration
    - `setpoints_vs_outdoor.png` and `pso_convergence.png`: plots for setpoints and convergence
- An overall convergence plot for all runs is saved as `results/convergence_all_runs.png`.

---

## 🔧 PSO Parameters in the Streamlit Interface

```python
num_particles = st.sidebar.slider("Number of Particles", 5, 100, 40)
iterations = st.sidebar.slider("Number of Iterations", 10, 300, 200)
w = st.sidebar.slider("Inertia Weight (w)", 0.1, 1.0, 0.7)
c1 = st.sidebar.slider("Cognitive Coefficient (c1)", 0.1, 3.0, 1.5)
c2 = st.sidebar.slider("Social Coefficient (c2)", 0.1, 3.0, 1.5)
```

### Default Values Explanation:
- num_particles = 40: a moderate number of particles to balance accuracy and runtime
- iterations = 200: a suitable number of iterations to reach convergence for this simple model
- w = 0.7: inertia weight to retain previous velocity without excessive momentum
- c1 = 1.5: influence of a particle's personal experience
- c2 = 1.5: influence of the swarm's best particle on others

> These default values were chosen based on initial experiments and common settings in standard PSO.

---

## 🔧 PSO Algorithm Type

- Particle Swarm Optimization (Global Best)
- Particles move according to three components:
    - Inertia Weight (w): retains previous velocity
    - Cognitive (c1): particle's personal experience
    - Social (c2): influence of the global best particle
- Suitable for simple nonlinear problems like optimizing building temperature within bounded ranges.

---

## 📚 Resources and Learning

- Kennedy, J., & Eberhart, R. (1995). Particle Swarm Optimization. Proceedings of ICNN.
- https://docs.streamlit.io
- Books and papers on building energy optimization
