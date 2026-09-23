# EV Adoption Model (LLM-Driven Agent-Based Simulation)

## Overview

The **EV Adoption Model** simulates how households decide whether to adopt **Electric Vehicles (EV)** or keep **Internal Combustion Engine (ICE)** vehicles.

The model combines **Agent-Based Modeling (ABM)** with **LLM reasoning** using the **mesa-llm framework**.

Each household evaluates **economic, environmental, and infrastructure factors** to determine the **utility of adopting an EV versus an ICE vehicle**. Based on this evaluation, agents use tools to purchase vehicles or charge their EVs.

This model demonstrates how **LLM agents can interact with structured decision models**, where economic formulas determine utilities while the LLM explains and executes actions.

---

# Model Architecture

The simulation consists of two main agent types.

---

## 1. HouseholdAgent

Represents a household making transportation decisions.

Households evaluate:

* Vehicle cost
* Infrastructure availability
* Environmental awareness
* Social influence
* Risk perception

Households may perform the following actions:

* Purchase an EV
* Purchase an ICE vehicle
* Charge an EV battery

### Household Agent Attributes

| Attribute      | Description                      |
| -------------- | -------------------------------- |
| income         | Annual household income          |
| env_awareness  | Environmental concern level      |
| annual_mileage | Expected yearly driving distance |
| state          | Vehicle ownership state          |
| utility_ev     | Utility score for EV             |
| utility_ice    | Utility score for ICE            |

### Vehicle States

```bash
NONE_HOLDER
EV_HOLDER
ICE_HOLDER
```

---

## 2. ChargingStationAgent

Represents EV charging infrastructure.

Charging stations include:

| Attribute        | Description                   |
| ---------------- | ----------------------------- |
| capacity         | Maximum simultaneous vehicles |
| price_per_kwh    | Electricity price             |
| charging_speed   | Energy delivered per step     |
| utilization_rate | Current usage                 |

Charging stations influence EV adoption by affecting the **infrastructure score in the utility function**.

---

# Decision Model

Households compute utilities for **EV** and **ICE** vehicles.

### EV Utility Function

```bash
U_EV = αF + βS + γI + δE − θR
```

Where:

| Variable | Meaning                     |
| -------- | --------------------------- |
| F        | Financial attractiveness    |
| S        | Social influence            |
| I        | Infrastructure availability |
| E        | Environmental motivation    |
| R        | Risk perception             |

### Adoption Decision Rule

```bash
If U_EV > U_ICE -> adopt EV
Else -> adopt ICE
```

---

# Financial Cost Model

Total cost of vehicle ownership is computed as follows.

### ICE Vehicle Cost

```bash
Total_ICE_Cost =
purchase_price_ice
+ fuel_price * annual_mileage / fuel_efficiency
+ maintenance_ice
```

### EV Vehicle Cost

```bash
Total_EV_Cost =
purchase_price_ev
- subsidy
+ electricity_price * annual_charge / ev_efficiency
+ maintenance_ev
```

The **difference between EV and ICE cost contributes to the financial utility component**.

---

# Infrastructure Model

Charging infrastructure is evaluated using **distance and congestion**.

```bash
Infrastructure = 1 / (1 + distance_to_station + congestion_penalty)
```

Where:

```bash
congestion_penalty = utilization_rate / capacity
```

Higher infrastructure scores increase **EV adoption probability**.

---

# LLM Agent Reasoning

Agents use the **ReAct reasoning framework** provided by **mesa-llm**.

Each simulation step:

1. Observe environment
2. Evaluate utilities
3. Generate reasoning
4. Select appropriate tool

### Example Reasoning Output

```bash
EV utility = 0.18
ICE utility = 0.11

EV utility is higher
Therefore purchasing an EV is the rational choice
```

---

# Available Tools

The following tools allow agents to interact with the environment.

## buy_ev

Purchases an electric vehicle.

Condition:

```bash
Agent does not already own a vehicle
```

---

## buy_ice

Purchases an internal combustion engine vehicle.

Condition:

```bash
Agent does not already own a vehicle
```

---

## charge_ev

Charges an EV battery at the nearest charging station.

Condition:

```bash
Agent owns an EV
Battery level is low
```

---

# Simulation Workflow

Each simulation step:

1. Agents observe the environment
2. Utilities are calculated
3. LLM reasoning explains the decision
4. Agents call tools when needed
5. Model collects system data

The simulation records:

* EV adoption rate
* ICE vehicle count
* Charging station utilization

---

# Visualization

The model uses **Solara visualization**.

Agents appear on a grid:

| Color | Meaning          |
| ----- | ---------------- |
| Green | EV household     |
| Red   | ICE household    |
| Blue  | Charging station |

Charts display:

* EV adoption over time
* ICE vehicle count

---

# File Structure

```bash
ev_model/
│
├── agent.py
├── tools.py
├── model.py
├── app.py
└── README.md
```

| File      | Purpose                                         |
| --------- | ----------------------------------------------- |
| agent.py  | Defines HouseholdAgent and ChargingStationAgent |
| tools.py  | Tool implementations                            |
| model.py  | Core simulation model                           |
| app.py    | Solara visualization                            |
| README.md | Project documentation                           |

---

# Running the Model

## Install Dependencies

```bash
pip install mesa
pip install mesa-llm
pip install solara
pip install python-dotenv
```

---

## Optional: LLM Provider

Create `.env` file:

```bash
OPENAI_API_KEY=your_key_here
```

or use **Ollama locally**.

Example configuration:

```bash
llm_model="ollama/llama3.1:latest"
```

---

## Run the Visualization

Navigate to the model folder:

```bash
cd examples/ev_model
```

Run the simulation:

```bash
solara run app.py
```

Open the interface in your browser:

```bash
http://localhost:8765
```

---

# Research Motivation

Electric vehicle adoption depends on multiple interacting factors:

* Socioeconomic heterogeneity
* Charging infrastructure availability
* Social diffusion
* Environmental awareness

Traditional equation-based models struggle to capture these dynamics.

Agent-based modeling enables the study of:

* adoption cascades
* spatial clustering
* infrastructure constraints
* policy experiments

---

# Possible Extensions

Future improvements may include:

* Government policy agents
* Subsidy experiments
* Real geographic infrastructure using mesa-geo
* Social network diffusion models
* Calibration with real EV adoption data

---


