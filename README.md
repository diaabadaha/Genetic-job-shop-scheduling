# Genetic Job Shop Scheduling

This project is part of the Artificial Intelligence course (ENCS3340) at Birzeit University. It focuses on optimizing job shop scheduling in a manufacturing plant using a genetic algorithm as a local search method.

## Project Description

In a manufacturing plant with several machines such as cutting machines, drilling machines, and assembly stations, each product requires a sequence of operations on these machines. The scheduling problem is to determine the optimal sequence and timing for each product to minimize the overall production time or maximize throughput while considering machine capacities and job dependencies.

This project aims to develop a genetic algorithm to optimize job shop scheduling in a manufacturing plant setting.

## Features

- **Input Handling**: The system takes as input a list of jobs and the number of available machines. Each job is defined as a sequence of operations, where each operation is specified by a machine to perform the task and the required processing time.
- **Scheduling Output**: The system outputs a schedule for each machine that depicts the start and end time for each process and to which job it belongs. A Gantt chart is used for this purpose.
- **Genetic Algorithm**: The algorithm uses chromosome representation, cross-over, mutation, and an objective function to find the optimal job scheduling.

## Usage

1. **Input Format**: 
   - Job_1: M1[10] -> M2[5] -> M4[12]
   - Job_2: M2[7] -> M3[15] -> M1[8]

2. **Running the Code**: 
   - Ensure you have Python installed.
   - Run the script `Genetic_job_scheduling.py` with the appropriate inputs.

3. **Output**: 
   - The script will generate a Gantt chart showing the scheduling results.

## Files

- `Genetic_job_scheduling.py`: The main code file implementing the genetic algorithm for job shop scheduling.
- `Project_description`: The main requirements of the project.

## Getting Started

### Prerequisites

- Python 3.x


