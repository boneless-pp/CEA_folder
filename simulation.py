import ctypes
import numpy as np
import csv
import matplotlib.pyplot as plt
import random
from scipy.sparse import lil_matrix, csr_matrix, find
import time


# Set seed for reproducibility
np.random.seed(42)
random.seed(42)


# Define the Person class
class Person:
    def __init__(self, id):
        self.infected = False
        self.neighbors = []
        self.id = id
        self.link = 0
        self.vax = False
        self.nextInfected = False
        self.recovered = False

    def add_neighbor(self, neighborId):
        self.neighbors.append(neighborId)
        self.link += 1

    def seek(self):
        self.infected = True

    def will_be_seek(self):
        self.nextInfected = True

    def vaccined(self):
        self.vax = True

    def recover(self):
        self.recovered = True


# Define the simulation functions
def are_you_infected(proba):
    result = np.random.binomial(1, proba)
    return result

def are_you_vaccined(proba):
    result = np.random.binomial(1, proba)
    return result

def are_you_recover(proba):
    result = np.random.binomial(1, proba)
    return result
    

# Load the shared library
libpagerank = ctypes.CDLL('./libpagerank.so')

class Edge(ctypes.Structure):
    _fields_ = [("from", ctypes.c_int), ("to", ctypes.c_int)]
    
# Define the return type and argument types for the calculate_pagerank function
libpagerank.calculate_pagerank.restype = None
libpagerank.calculate_pagerank.argtypes = [ctypes.POINTER(Edge), ctypes.c_longlong, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]



def call_pagerank(edges, nodes):
    edges_array = (Edge * len(edges))(*edges)
    pagerank_scores = (ctypes.c_double * nodes)()
    libpagerank.calculate_pagerank(edges_array, len(edges), nodes, pagerank_scores)
    return np.array(pagerank_scores[:])

def init_simulation(size, mode, vax_proba, infection_proba, page_rank_vector, initial_infected=10):
    population = [Person(i) for i in range(size)]
    
    if mode == "smart":
        num_to_vaccinate = int(0.7 * size)  # Vaccinate the top 70% based on PageRank
        top_indices_to_vaccinate = np.argsort(page_rank_vector)[-num_to_vaccinate:]
        for i in top_indices_to_vaccinate:
            population[i].vaccined()

    # Infect a specific number of individuals initially, regardless of vaccination status
    initial_infected_indices = np.random.choice(range(size), initial_infected, replace=False)
    for i in initial_infected_indices:
        population[i].seek()

    # Random vaccination for the rest of the population if mode is "random"
    for person in population:
        if mode == "random" and not person.infected and are_you_vaccined(vax_proba):
            person.vaccined()

    return population


def simulation_step(population, matrix, recovery_proba, infection_proba):
    # Iterate over each person in the population
    for person in population:
        if person.infected:
            # Infect neighbors with a certain probability
            start_ptr = matrix.indptr[person.id]
            end_ptr = matrix.indptr[person.id + 1]
            neighbors = matrix.indices[start_ptr:end_ptr]
            for neighbor_id in neighbors:
                neighbor = population[neighbor_id]
                if not neighbor.vax and not neighbor.recovered and not neighbor.infected and are_you_infected(infection_proba):
                    neighbor.will_be_seek()

            # Recovery process
            if are_you_recover(recovery_proba):
                person.recover()

    # Update status for all population members
    for person in population:
        if person.nextInfected:
            person.seek()
            person.nextInfected = False



def simulation_loop(population, matrix, num_steps, recovery_proba, infection_proba):
    # Main simulation loop
    number_of_infected = []
    for step in range(num_steps):
        simulation_step(population, matrix, recovery_proba, infection_proba)
        infected_count = sum(person.infected for person in population)
        recovered_count = sum(person.recovered for person in population)
        vaccinated_count = sum(person.vax for person in population)
        
        # Print the current statistics
        print(f"Step {step}: Infected: {infected_count}, Recovered: {recovered_count}, Vaccinated: {vaccinated_count}")
        
        number_of_infected.append(infected_count)
        
        if infected_count == 0:
            print(f"The epidemic has ended at step {step}.")
            break
    
    return number_of_infected
    


def get_data():
    data = []
    max_id = 0  # Initialize to keep track of the maximum node ID
    with open('./mapped_twitter_combined.txt') as f:
        for item in f:
            parts = item.split()
            from_id, to_id = int(parts[0]), int(parts[1])
            max_id = max(max_id, from_id, to_id)  # Update max_id if necessary
            data.append([from_id, to_id])
    print(f"Maximum node ID in data: {max_id}")
    return data, max_id


def init_matrix_from_mapped_data(filepath, size):
    # Initialize an empty LIL matrix for efficient element-wise operations
    matrix = lil_matrix((size, size), dtype=int)
    with open(filepath, 'r') as file:
        for line in file:
            from_node, to_node = map(int, line.split())
            matrix[from_node, to_node] = 1
            matrix[to_node, from_node] = 1  # Assuming undirected graph

    # Convert to CSR format for efficient mathematical operations afterward
    return matrix.tocsr()


# Parameters
edges_data, max_id = get_data()
edges = [(from_node, to_node) for from_node, to_node in edges_data]
size = max_id + 1  # Adjusted to the max node id in the dataset
infection_proba = 0.05  # Adjusted infection rate
recovery_proba = 0.1  # Recovery rate
vax_proba_random = 0.4  # Vaccination rate for random vaccination
vax_proba_none = 0  # Vaccination rate for no vaccination
vax_proba_smart = 0.70  # or 70%


matrix = init_matrix_from_mapped_data('./mapped_twitter_combined.txt', size)


# Compute PageRank vector  
edges_ctypes = (Edge * len(edges))(*[Edge(from_node, to_node) for from_node, to_node in edges])
start_time = time.time()
page_rank_vector = call_pagerank(edges_ctypes, size)
end_time = time.time()
print(f"PageRank calculation took {end_time - start_time} seconds.")
sorted_indices = np.argsort(page_rank_vector)[::-1]
page_rank_vector_sorted = page_rank_vector[sorted_indices]

# Initialize the simulations with no vaccination, random vaccination, and smart vaccination
population_no_vax = init_simulation(size, "no_vax", vax_proba_none, infection_proba, page_rank_vector_sorted)
population_random = init_simulation(size, "random", vax_proba_random, infection_proba, page_rank_vector_sorted)
population_smart = init_simulation(size, "smart", vax_proba_smart, infection_proba, page_rank_vector_sorted)

# Run the simulations
num_steps = 40
start_time = time.time()
infected_counts_no_vax = simulation_loop(population_no_vax, matrix, num_steps, recovery_proba, infection_proba)
end_time = time.time()
print(f"Simulation loop (no vaccination) took {end_time - start_time} seconds.")
start_time = time.time()
infected_counts_random = simulation_loop(population_random, matrix, num_steps, recovery_proba, infection_proba)
end_time = time.time()
print(f"Simulation loop (random) took {end_time - start_time} seconds.")
start_time = time.time()
infected_counts_smart = simulation_loop(population_smart, matrix, num_steps, recovery_proba, infection_proba)
end_time = time.time()
print(f"Simulation loop (smart) took {end_time - start_time} seconds.")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(infected_counts_no_vax, label='No Vaccination', linewidth=2)
plt.plot(infected_counts_random, label='Random Vaccination', linewidth=2)
plt.plot(infected_counts_smart, label='Smart Vaccination', linewidth=2)
plt.xlabel('Time Steps')
plt.ylabel('Number of Infected Individuals')
plt.yscale('log')
plt.title('Simulation of Epidemic Spread')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('plt_infected_time4.png')
