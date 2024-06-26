
# An molecular memory similar to "Reinvent + scaffold memory" is needed, 
# in which the molecular fingerprints are stored instead of the scaffolds
memory = pd.DataFrame(columns=["smiles", "scores", "fps"])

def memory_update(smiles, scores):
    for i in range(len(smiles)):

        fp = calc_fingerprints([smiles[i]])
        dists = calc_dists(fp, memory["fps"])

        # Adding the diversity enhancement term
        scores[i] += sigma1 * np.min(dists)

        # Update the memory
        if scores[i] > threshold:
            new_data = pd.DataFrame({"smiles": smiles[i], "scores": scores[i], "fps": fp})
            memory = pd.concat([memory, new_data], ignore_index=True, sort=False)

        return smiles, scores