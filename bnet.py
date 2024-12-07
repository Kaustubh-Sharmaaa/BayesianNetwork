#Name: Kaustubh Sharma
#UTA ID: 1002138514

from collections import defaultdict
from itertools import product

# Function to read training data from a file
def input_training_data(filepath):
    with open(filepath, "r") as file:
        return [[int(value) for value in line.strip().split()] for line in file]

# Function to tally counts for Bayesian network computation
def tally_counts(dataset):
    tally = {
        "B": defaultdict(int),
        "G|B": defaultdict(int),
        "C": defaultdict(int),
        "F|G,C": defaultdict(int),
    }

    for entry in dataset:
        B, G, C, F = entry
        tally["B"][(B,)] += 1
        tally["G|B"][(G, B)] += 1
        tally["C"][(C,)] += 1
        tally["F|G,C"][(F, G, C)] += 1

    return tally

# Calculate Conditional Probability Tables (CPTs)
def computation(dataset, counts):
    cpts = {
        "B": {},
        "G|B": {},
        "C": {},
        "F|G,C": {},
    }

    num_samples = len(dataset)

    # Calculate CPT for B
    for (B,), cnt in counts["B"].items():
        cpts["B"][B] = cnt / num_samples

    # Calculate CPT for G given B
    for (G, B), cnt in counts["G|B"].items():
        sum_B = sum(counts["G|B"][(g, B)] for g in [0, 1])
        cpts["G|B"][(G, B)] = cnt / sum_B

    # Calculate CPT for C
    for (C,), cnt in counts["C"].items():
        cpts["C"][C] = cnt / num_samples

    # Calculate CPT for F given G and C
    for (F, G, C), cnt in counts["F|G,C"].items():
        sum_GC = sum(counts["F|G,C"][(f, G, C)] for f in [0, 1])
        cpts["F|G,C"][(F, G, C)] = cnt / sum_GC

    return cpts

# Function to calculate joint probability for an event
def calculate_joint_probability(event, cpts):
    probability = 1.0
    probability *= cpts["B"][event["B"]]
    probability *= cpts["G|B"][(event["G"], event["B"])]
    probability *= cpts["C"][event["C"]]
    probability *= cpts["F|G,C"][(event["F"], event["G"], event["C"])]
    return probability

# Function to sum over all possible values of hidden variables
def sum_over_hidden(variables, observed, cpts):
    hidden_vars = [v for v in variables if v not in observed]
    possible_values = product([0, 1], repeat=len(hidden_vars))
    total_probability = 0.0

    for values in possible_values:
        full_event = observed.copy()
        for i, var in enumerate(hidden_vars):
            full_event[var] = values[i]
        total_probability += calculate_joint_probability(full_event, cpts)

    return total_probability

# Bayesian query function
def perform_query(query_vars, given_vars, cpts):
    variables = ["B", "G", "C", "F"]
    full_event = {**query_vars, **given_vars}
    numerator = sum_over_hidden(variables, full_event, cpts)
    denominator = sum_over_hidden(variables, given_vars, cpts)
    return numerator / denominator

# Parse user queries
def interpret_query(query_str):
    values = {"t": 1, "f": 0}
    query = {}
    parts = query_str.split()
    index = 0
    while index < len(parts):
        part = parts[index]
        var_name = part[0]
        var_value = values[part[1]]
        query[var_name] = var_value
        index += 1
    return query

# Interactive user input
def input_loop(cpts):
    while True:
        user_query = input("Query: ").strip()
        if user_query.lower() == "bye":
            print("Exiting.")
            break

        if "given" in user_query:
            query_section, evidence_section = user_query.split("given")
            query_variables = interpret_query(query_section.strip())
            evidence_variables = interpret_query(evidence_section.strip())
        else:
            query_variables = interpret_query(user_query.strip())
            evidence_variables = {}

        probability = perform_query(query_variables, evidence_variables, cpts)
        print(f"Probability: {probability:.3f}")

def main():
    bayesian_network = {
        "B": {"parents": [], "cpt": {}},
        "G": {"parents": ["B"], "cpt": {}},
        "C": {"parents": [], "cpt": {}},
        "F": {"parents": ["G", "C"], "cpt": {}},
    }
    training_file_path = "training_data.txt"
    training_data = input_training_data(training_file_path)
    data_counts = tally_counts(training_data)
    computed_cpts = computation(training_data, data_counts)
    print("Conditional Probability Tables (CPTs):")
    for var, table in computed_cpts.items():
        print(f"\n{var}:")
        for condition, probability in table.items():
            print(f"{condition}: {probability:.3f}")
    input_loop(computed_cpts)

if __name__ == "__main__":
    main()
