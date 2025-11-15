import numpy as np
import matplotlib.pyplot as plt
from Bio import pairwise2  # For performing local alignment


def find_local_alignment_segments(genome1, genome2, k):
    alignment_segments = []

    for i in range(len(genome1)):
        start_index = max(0, i - k)
        end_index = min(len(genome1), i + k + 1)
        segment_genome1 = genome1[start_index:end_index]

        alignments = pairwise2.align.localms(
            segment_genome1, genome2, 4, -1.5, -3.5, -2.5
        )
        alignment_genome2 = alignments[0][1]
        alignment_segment_genome2 = alignment_genome2[
            alignments[0][3] : alignments[0][3] + alignments[0][4]
        ]
        alignment_segments.append((segment_genome1, alignment_segment_genome2))

    return alignment_segments


def calculate_k_si_distribution(alignment_segments):
    k_si_list = []

    for segment_genome1, alignment_segment_genome2 in alignment_segments:
        k_si_score = sum(alignment_segment_genome2.count(n) for n in segment_genome1)
        k_si_list.append(k_si_score)

    return k_si_list


def calculate_probability_distribution(k_si_list):
    probabilities = [count / sum(k_si_list) for count in k_si_list]
    return probabilities


def plot_probability_distribution(probabilities):
    # Generate x-values (positions)
    positions = np.arange(1, len(probabilities) + 1)

    # Plot the probability distribution
    plt.plot(positions, probabilities, marker="o", linestyle="-")
    plt.xlabel("Position")
    plt.ylabel("Probability")
    plt.title("Probability Density Function (PDF)")
    plt.grid(True)
    plt.show()


# Sample Sequences
genome1 = "ACGCACGTGATCGATCGATCGATCGGGATT"
genome2 = "TCCTACGGGGCTGACAAATATATCTGATCTT"
k = 5

alignment_segments = find_local_alignment_segments(genome1, genome2, k)
k_si_list = calculate_k_si_distribution(alignment_segments)
probabilities = calculate_probability_distribution(k_si_list)
# print(alignment_segments)

print("K-Synteny Index List for Genome1:")
for i, k_si_score in enumerate(k_si_list):
    print(f"Position {i+1}: {k_si_score}")

print("\nProbability Distribution for Genome1 (Based on Count):")
for i, probability in enumerate(probabilities):
    print(f"Position {i+1}: {probability}")

plot_probability_distribution(probabilities)
