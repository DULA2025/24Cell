# Prime Number Graph Visualization and Hadwiger-Nelson Problem Exploration

## Overview

This project explores the intersection of number theory, graph theory, and geometric visualization by constructing and analyzing graphs based on prime numbers congruent to 1 or 5 modulo 6. The connectivity of the graph is defined with geometric considerations, providing a unique perspective on graph structure and its relation to the Hadwiger-Nelson problem.

## Accomplishments

- **Graph Construction Based on Prime Numbers**: We've created a graph where each vertex corresponds to prime numbers congruent to 1 or 5 mod 6, blending number theory with graph theory.
  
- **Geometric Connectivity Rules**: The edges are determined by rules that simulate geometric properties:
  - Connecting vertices if the difference between their prime indices is prime.
  - Using multiples, perfect squares, Pythagorean triples, and angle relationships to mimic known geometric structures.

- **Scaling Up**: Starting from smaller graphs, we scaled to 72 vertices, enhancing complexity while preserving our prime number and geometric rules.

- **3D Visualization with CUDA and JIT**: Implemented a 3D visualization using PyVista, with CUDA for potential GPU acceleration and Numba's JIT compilation for performance in prime calculations and graph construction.

- **CNF Formula for Graph Coloring**: Generated a CNF formula to tackle the graph coloring problem, useful for determining the chromatic number with SAT solvers.

## Connection to the Hadwiger-Nelson Problem

The Hadwiger-Nelson problem seeks to find the chromatic number of the plane, **χ(ℝ²)**, which is the minimum number of colors needed to color points so that no two points at unit distance share the same color. Here's how our project relates:

- **Graph as a Model**: Our graph acts as a discrete model of the plane, with vertices as points and edges as unit or geometrically defined distances.

- **Prime Number Constraints**: By using primes congruent to 1 or 5 mod 6, we introduce a structured distribution, offering insights into how such constraints affect coloring.

- **Geometric Rules**: These rules simulate geometric relationships, reflecting how configurations in the plane might influence its chromatic number.

- **Chromatic Number Exploration**: The CNF formula generation and potential solving mimic the exploration of the plane's chromatic number through a finite, complex graph.

- **Theoretical Insights**: Our project provides theoretical insights into chromatic number lower bounds by examining graph complexity under our defined rules.

## Usage

To run this project:

1. Ensure you have Python 3.11 or later installed with `numpy`, `numba`, and `pyvista` libraries.
2. Execute the script provided in the repository. The script will:
   - Generate a graph based on prime numbers.
   - Create an adjacency matrix for the graph.
   - Output a CNF formula for graph coloring.
   - Visualize the graph in 3D using PyVista.

## Future Directions

- **Subgraph Analysis**: Investigate subgraphs to uncover simpler geometric patterns or relate them to known structures.
- **Interactive Visualization**: Enhance visualization with interactive elements for deeper exploration.
- **Graph Coloring Solutions**: Use SAT solvers to explore the chromatic number or color distribution patterns.
- **Theoretical Expansion**: Continue exploring theoretical implications regarding graph properties influenced by prime numbers and geometry.

This project not only combines different mathematical fields but also serves as a practical exploration of complex graph theory problems like the Hadwiger-Nelson problem, offering insights into how prime numbers and geometric constraints might influence graph coloring in abstract spaces.
