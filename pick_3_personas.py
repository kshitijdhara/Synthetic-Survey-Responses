from datasets import load_dataset
import random

# Load the dataset
print("Loading the dataset...")
ds = load_dataset("proj-persona/PersonaHub", "persona")

# Check if the dataset has a 'train' split
if 'train' not in ds:
    print("Error: The dataset does not have a 'train' split.")
    exit()

# Get the total number of personas
total_personas = len(ds['train'])
print(f"Total number of personas: {total_personas}")

# Randomly select 3 indices
selected_indices = random.sample(range(total_personas), 10)

# Print the selected personas
for i, index in enumerate(selected_indices, 1):
    persona = ds['train'][index]
    print(f"\nPersona {i}:")
    print(persona['persona'])
