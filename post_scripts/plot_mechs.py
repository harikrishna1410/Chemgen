import os
import cantera as ct
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Function to get mechanism stats from yaml file
def get_mech_stats(yaml_file):
    try:
        gas = ct.Solution(yaml_file)
        return {
            'n_species': gas.n_species,
            'n_reactions': gas.n_reactions
        }
    except Exception as e:
        print(f"Error processing {yaml_file}: {str(e)}")
        return None


def make_mech_summary(yaml_files):
    # Dictionary to store metadata
    metadata = {}

    # Process each yaml file
    for yaml_file in yaml_files:
        basename = os.path.basename(yaml_file).split('.')[0]  # Get filename without path and extension
        stats = get_mech_stats(yaml_file)
        if stats:
            metadata[basename] = stats
            metadata[basename]['yaml_file'] = yaml_file

    return metadata

# Find all yaml files
yaml_files = [str(p) for p in Path('../ext_repo').glob('**/*.yaml')]

if os.path.exists('mech_stats.json'):
    import json
    with open('mech_stats.json', 'r') as f:
        metadata = json.load(f)
else:
    metadata = make_mech_summary(yaml_files)
    import json
    with open('mech_stats.json', 'w') as f:
        json.dump(metadata, f)

# Prepare data for plotting
mechanisms = list(metadata.keys())
n_species = [metadata[m]['n_species'] for m in mechanisms]
n_reactions = [metadata[m]['n_reactions'] for m in mechanisms]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot number of species
for i, mech in enumerate(mechanisms):
    ax1.scatter(i, n_species[i], alpha=0.6,label=mech)
    # ax1.annotate(mech, (i, n_species[i]), xytext=(5, 5), textcoords='offset points')
ax1.set_ylim(0, 100)
# ax1.legend()
ax1.set_xlabel('Mechanism Index')
ax1.set_ylabel('Number of Species')
ax1.set_title('Number of Species per Mechanism')
ax1.grid(True)

# Plot number of reactions
for i, mech in enumerate(mechanisms):
    ax2.scatter(i, n_reactions[i], alpha=0.6, label=mech)
    # ax2.annotate(mech, (i, n_reactions[i]), xytext=(5, 5), textcoords='offset points')
ax2.set_ylim(0, 1000)
ax2.set_xlabel('Mechanism Index')
ax2.set_ylabel('Number of Reactions')
ax2.set_title('Number of Reactions per Mechanism')
ax2.grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

# Print summary
print("\nMechanism Summary:")
for mech in mechanisms:
    print(f"{mech}:")
    print(f"  Species: {metadata[mech]['n_species']}")
    print(f"  Reactions: {metadata[mech]['n_reactions']}")