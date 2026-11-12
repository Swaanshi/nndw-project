# Carbon-Aware Traffic Forecasting and Energy-Efficient Sleep Scheduling

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![NS-3](https://img.shields.io/badge/NS--3-3.41-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

This repository contains the code and implementation details for the research paper: **"Carbon-Aware Traffic Forecasting and Energy-Efficient Sleep Scheduling for Sustainable Communication Networks"** by Arul Jain and Swaanshi Jain.

## 📖 Overview

The rapid expansion of telecommunication infrastructure has significantly increased energy consumption and grid carbon emissions. This project presents a multi-objective framework for carbon-aware network operation that implements proactive traffic prediction coupled with adaptive sleep scheduling. 

By leveraging historical mobile network activity (Telecom Italia Cellular Dataset) and electricity grid carbon intensity measurements (Electricity Maps), this system achieves a **45.55% reduction in total network energy consumption and grid carbon emissions** while maintaining strict Quality of Service (QoS) with packet loss rates strictly below 1%.

### Core Components
1. **Prediction Layer:** A multi-layer Long Short-Term Memory (LSTM) deep learning model that forecasts localized traffic demand and grid carbon intensity over a spatio-temporal window.
2. **Optimization Layer:** A Multi-Objective Optimization (MOO) decision engine that dynamically transitions network nodes between *Active*, *Light Sleep*, and *Deep Sleep* hardware states based on real-time traffic demand and localized solar microgrid availability.
3. **Simulation Layer (Digital Twin):** An NS-3 based digital twin of a 30-node urban topology (Milan) governed by the Optimized Link State Routing (OLSR) protocol and strict priority queuing (PfifoFastQueueDisc) to protect latency-sensitive (VIP) traffic during hardware throttling.

## ⚙️ Prerequisites

To run the models and simulations, you will need:
* **Python 3.8+** (for the LSTM prediction model and data processing)
* **NS-3 (Version 3.41)** (for the discrete-event network simulation)
* A CUDA-enabled GPU (recommended for faster LSTM training via PyTorch)

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/carbon-aware-networking.git
   cd carbon-aware-networking
   ```

2. **Install Python Dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **NS-3 Simulation Setup:**
   * Download and build [ns-3.41](https://www.nsnam.org/releases/ns-3-41/).
   * Copy the simulation scripts from this repository's `ns3_simulation/` folder (adjust path as needed based on your structure) into the `scratch/` directory of your NS-3 workspace.
   * Run the simulation via waf/ns3:
     ```bash
     ./ns3 run scratch/your_simulation_script
     ```

## 📊 Datasets

The models are trained using two primary data sources:
* **Traffic Workloads:** Telecom Italia Cellular Dataset (available via Harvard Dataverse).
* **Grid Emissions:** Hourly $gCO_2eq/kWh$ data from the Electricity Maps platform.

*(Note: Due to file size limits, the raw datasets are not included in this repository. Please download them from their respective sources and place them in the `data/` directory.)*

## 📜 Citation

If you find this code or research helpful in your work, please consider citing our paper:
```bibtex
@article{Jain2023CarbonAware,
  title={Carbon-Aware Traffic Forecasting and Energy-Efficient Sleep Scheduling for Sustainable Communication Networks},
  author={Jain, Arul and Jain, Swaanshi},
  institution={Netaji Subhas University of Technology, New Delhi},
  year={2023}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.