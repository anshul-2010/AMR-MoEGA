"""
Download script for MicroBIGG-E or other genome datasets.
"""
from Bio import Entrez
from Bio import SeqIO

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/content/drive/MyDrive/AMR_Prediction/microbigge.csv")

df = df.drop(
    columns=[
        "BioSample",
        "Strain",
        "Collection date",
        "Assembly",
        "Isolate",
        "Contig",
        "Strand",
        "Element symbol",
        "Type",
        "Scope",
        "Subtype",
        "Method",
        "Serovar",
        "% Coverage of reference",
        "% Identity to reference",
    ]
)
df.dropna(inplace=True)
df = df[df["Organism group"] != "Vibrio metschnikovii"]
df = df[df["Organism group"] != "Yersinia ruckeri"]

sampled_data, required_df = train_test_split(
    df, test_size=5000, stratify=df["Organism group"]
)
sampled_data.reset_index(drop=True, inplace=True)
required_df.reset_index(drop=True, inplace=True)

Entrez.email = "Temporary Email ID"
sequences = []
for i in range(len(required_df)):
    protein_id = required_df["Protein"][i]
    handle = Entrez.efetch(db="protein", id=protein_id, rettype="fasta", retmode="text")
    sequence_record = SeqIO.read(handle, "fasta")
    handle.close()
    sequences.append(str(sequence_record.seq))

df2 = required_df.assign(sequence=sequences)
df2.to_csv("/content/drive/MyDrive/AMR_Prediction/Sequence_Data.csv", index=False)
