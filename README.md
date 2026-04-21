# 2E-EVRP Algorithm Comparison
## Project Overview
This project implements and compares four different algorithms for tackling the **Two-Echelon Electric Vehicle Routing Problem (2E-EVRP)**, including the Clarke-Wright savings algorithm and two evolutionary algorithms.

## Problem Description
The Two-Echelon Electric Vehicle Routing Problem (2E-EVRP) is a variation of the classic NP-hard Electric Vehicle Routing problem. In the 2E-EVRP, there are two vehicle types: larger vehicles (LVs) that transport goods in bulk from a central depot to intermediate facilities called hubs, and smaller electric vehicles (EVs) that pick up goods from these hubs and deliver them to customers. Our objective is to find the minimum total operation cost for the entire process.

## Algorithms Used
### 1. Brute Force (Exact Solution)
- **Time Complexity**: Θ(n * n!)
- **Concept**: This approach exhaustively searches every possible route that can be taken by every vehicle. It starts by finding the minimum number of EVs needed for a solution to exist, and once its found that, it then tries to find the optimal solution for that number of EVs (the one that minimizes distance traveled).
- **Strengths**: always finds the optimal (exact) solution.
- **Weaknesses**: infeasible for any input size greater than five customers due to a combinatorial explosion when it comes to the number of routes that need to be searched.
- **Why Include This**: for small input sizes, we can test to see how close our approximate solutions are to the true solution, making brute force a good baseline for small n.
### 2. Clarke-Wright Savings Algorithm (Standard)
- **Time Complexity**: Θ(n^2 log n)
- **Concept**: This algorithm finds one or more routes that minimize a given cost function by first starting with a bunch of naive/direct (and inefficient) routes and iteratively combining them in such a way that savings (the difference between the current cost and the new cost) is greedily maximized until no more routes can be combined without increasing costs. 
- **Strengths**: fast/efficient at finding a locally optimal solution.
- **Weaknesses**: not guaranteed to find a globally optimal solution.
- **Why Include This**: the Clarke-Wright Savings Algorithm is a classic algorithm that has been widely adopted for decades.
### 3. Adaptive Large Neighborhood Search
- **Time Complexity**: Θ(n^2), or Θ(m * n^2) if you don’t consider the number of iterations a constant. 
- **Concept**: This algorithm first starts with a random solution, and then iteratively attempts to improve it by destroying parts of it and then reconstructing it.  This ensures that many solutions are tried while still being heuristically guided.
- **Strengths**: solution quality scales with additional time/compute.
- **Weaknesses**: not quite as direct as the Clarke-Wright Savings algorithm due to exploration of the solution space.
- **Why Include This**: adaptive large neighborhood search is able to strike a balance between direct (greedy) approaches like the Clarke-Wright Savings Algorithm and evolutionary approaches like the Memetic Algorithm by directly addressing the exploration-exploitation trade-off.
### 4. Memetic Algorithm (Nature-Inspired)
- **Time Complexity**: Θ(n^3) 
- **Concept**: In general, a memetic algorithm implements global search using populations (for us this will be routes) which are iterated evolutionarily, while implementing local search through refinements using a pre-determined "meme" (for us this will be based on constraints such as load capacity, battery life, etc.). The traits acquired through these refinements are then evaluated for fitness, with the traits of promising offspring being used to update the chromosomes for the next generation. 
- **Strengths**: fast if tuned correctly, adaptable.
- **Weaknesses**: tuning of parameters has a large impact on performance.
- **Why Include This**: it both satisfies the nature-inspired algorithm requirement and is a legitimately effective technique for TSP-style problems. It also gives the opportunity to inject some creativity into the algorithm through parameter tuning.

## Usage
To run, just do:
```
$ python3 code/main.py
```
The program will then prompt you to enter a file path (which should be a .txt file in data/) and to select which algorithm you want to use.

## File Structure
```
2E-EVRP_Clarke-Wright_vs_EA/
├── code/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   ├── evaluator.py
│   │   ├── parser.py
│   │   └── solver_runner.py
│   ├── gui/
│   │   ├── __init__.py
│   │   └── gui.py
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── memetic_helpers/
│   │   │   ├── adaptive_local_search.py
│   │   │   ├── backbone_crossover.py
│   │   │   └── k_pseudo_greedy.py
│   │   ├── brute_force.py
│   │   ├── clarke_wright.py
│   │   ├── memetic.py
│   │   └── neighborhood_search.py
│   └── main.py
├── data/
│   ├── Customer_5/
│   │   └── *.txt
│   ├── Customer_10/
│   │   └── *.txt
│   ├── Customer_15/
│   │   └── *.txt
│   ├── Customer_50/
│   │   └── *.txt
│   └── Customer_100/
│       └── *.txt
├── graphs/
├── references/
├── report/
├── slides/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Dataset
The dataset for this project can be found here: https://github.com/manilakbay/2E-EVRP-Instances/tree/main 
We use the Type_x instance and compare our algorithms on all input sizes (# customers: 5, 10, 15, 50, 100).

## Generative AI Usage Disclosure
The following AI tools were used to assist in writing code:
- o3
- GPT-5.4
- Claude Sonnet 4.6

## Contributors
- [Will Hemphill](https://github.com/will-hemphill)
- [Max Jessey](https://github.com/mjessey)
- [Preston Page](https://github.com/MaybePreston)
- [Braylon Trail](https://github.com/batrail)
