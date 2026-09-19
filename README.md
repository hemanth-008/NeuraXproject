# QualityTwin

> Quality-aware digital twin of a production line — Neurax 3.0 Hackathon, Domain 2 "AI in Industry and Automation"

## Problem Understanding

**Visual Inspection & Defect Root-Cause Assistant**: a software-only system that judges product quality, finds production bottlenecks, and estimates throughput, cost and profit, with advisory recommendations. Nothing connects to real hardware.

The system uses discrete-event simulation data from a shared manufacturing facility (Mendeley Data, Arena Simulation) to build a digital twin that can:
- Assess product quality and detect defects
- Identify root causes of production issues and bottlenecks
- Estimate throughput, cost, and profit
- Provide actionable recommendations

## Architecture

TODO

## Approach

TODO

## Data

Dataset: "Manufacturing Data Shared Facility — Discrete-Event Simulation" (Mendeley Data).
- 3 simulation models of increasing complexity (10, 16, 77 features)
- Covers: utilization, queue times, SKU counters, quality checks, VA/NVA time
- See `DATA_INVENTORY.md` for full inventory.

## Setup

```powershell
# Create virtual environment
python -m venv .venv

# Install dependencies
.venv\Scripts\pip.exe install -r requirements.txt

# Run data inventory
.venv\Scripts\python.exe src/ingest/inventory.py
```

## Limitations

- No real hardware connection — purely software simulation.
- Dataset is from Arena DES; real-world noise, sensor drift, image data are absent.
- No image/vision data in the provided dataset.

TODO

## References

- Rabaev, M., Pratama, M., & Chan, K. Y. (2019). *Manufacturing Data Shared Facility — Discrete-Event Simulation* [Data set]. Mendeley Data, v2. DOI: [10.17632/3rw227zxt7.2](https://doi.org/10.17632/3rw227zxt7.2). License: CC BY 4.0.
