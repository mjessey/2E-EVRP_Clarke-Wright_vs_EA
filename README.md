# 2E-EVRP Algorithm Comparison
## Project Overview
This project implements and compares four different algorithms for tackling the **Two-Echelon Electric Vehicle Routing Problem (2E-EVRP)**, including the Clarke-Wright savings algorithm and two evolutionary algorithms.

## Problem Description
The Two-Echelon Electric Vehicle Routing Problem (2E-EVRP) is a variation of the classic NP-hard Electric Vehicle Routing problem. In the 2E-EVRP, there are two vehicle types: larger vehicles (LVs) that transport goods in bulk from a central depot to intermediate facilities called hubs, and smaller electric vehicles (EVs) that pick up goods from these hubs and deliver them to customers. Our objective is to find the minimum total operation cost for the entire process.

## Algorithms Used
### 1. Brute Force (Exact Solution)
- **Time Complexity**:
- **Concept**:
- **Strengths**:
- **Weaknesses**:
- **Why Include This**:
### 2. Clarke-Wright Savings Algorithm (Baseline)
- **Time Complexity**:
- **Concept**:
- **Strengths**:
- **Weaknesses**:
- **Why Include This**:
### 3. Adaptive Large Neighborhood Search
- **Time Complexity**:
- **Concept**:
- **Strengths**:
- **Weaknesses**:
- **Why Include This**:
### 4. Memetic Algorithm (Nature-Inspired)
- **Time Complexity**:
- **Concept**:
- **Strengths**:
- **Weaknesses**:
- **Why Include This**:

## Usage
To run, just do:
```
$ python3 code/main.py
```
```
```

## File Structure
```
2E-EVRP_Clarke-Wright_vs_EA/
├── code/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── parser.py
│   ├── gui/
│   │   ├── __init__.py
│   │   └── gui.py
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── brute_force.py
│   │   └── clarke_wright.py
│   └── main.py
├── data/
│   ├── Customer_5/
│   │   └── *_C5x_n.txt
│   ├── Customer_10/
│   │   └── *_C10x_n.txt
│   ├── Customer_15/
│   │   └── *_C15x_n.txt
│   ├── Customer_50/
│   │   └── *_C50x_n.txt
│   └── Customer_100/
│       └── *_C100x_n.txt
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
The dataset for this project can be found here: https://github.com/manilakbay/2E-EVRP-Instances/tree/main <br \>
We use the Type_x instance and compare our algorithms on all input sizes (# customers: 5, 10, 15, 50, 100).

## Generative AI Usage Disclosure
The following AI tools were used to assist in writing code:
- o3
- Claude Sonnet 4.6

## Contributors
- [Will Hemphill](https://github.com/will-hemphill)
- [Max Jessey](https://github.com/mjessey)
- [Preston Page](https://github.com/MaybePreston)
- [Braylon Trail](https://github.com/batrail)
