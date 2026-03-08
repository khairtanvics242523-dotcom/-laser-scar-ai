import pandas as pd
import matplotlib.pyplot as plt

print("Loading dataset...")

data = pd.read_csv("patient_data.csv")

print("\nDataset Preview:\n")
print(data.head())

print("\nStatistics:\n")
print(data.describe())

reaction_counts = data["reaction"].value_counts()

print("\nReaction Distribution:\n")
print(reaction_counts)

print("Generating charts...")

# Chart 1: Reaction Distribution
plt.figure()
reaction_counts.plot(kind="bar")
plt.title("Scar Reaction Distribution")
plt.xlabel("Reaction Type")
plt.ylabel("Number of Patients")
plt.savefig("reaction_distribution.png")

# Chart 2: Age vs Reaction
plt.figure()
plt.scatter(data["age"], data["reaction"])
plt.title("Age vs Scar Reaction")
plt.xlabel("Age")
plt.ylabel("Reaction")
plt.savefig("age_vs_reaction.png")

print("Charts saved successfully!")