import pandas as pd
import numpy as np
import os

mj_list = [
    "MJ002", "MJ004", "MJ005", "MJ007", "MJ008", "MJ012", "MJ014", "MJ015",
    "MJ018", "MJ019", "MJ020", "MJ021", "MJ022", "MJ023",
]

mj_list = [
    "MJ005"
]

percentiles = [0.02, 0.05, 0.10, 0.25, 0.50]

for perc in percentiles:
    output_dir = f'binary_matrices_{perc*100:.0f}%'
    os.makedirs(output_dir, exist_ok=True)

    for MJ in mj_list:
        net_path = f'/shares/vasciaveo_lab/data/nepc_organoid_project/new_data/{MJ}/aracne3_output_tfs_and_cotfs_100subnet/consolidated-net_{MJ}.tsv'
        net = pd.read_csv(net_path, sep='\t')

        # Step 1: Create binary entries
        binary_entries = []

        for reg, group in net.groupby('regulator.values'):
            logp = group['log.p.values'].values
            cutoff = np.percentile(logp, perc*100)  # Top percentile
            group = group.copy()
            group['selected'] = (group['log.p.values'] <= cutoff).astype(int)
            binary_entries.extend(zip(group['regulator.values'], group['target.values'], group['selected']))

        # Step 2: Convert to binary matrix (rows = regulators, cols = targets)
        binary_df = pd.DataFrame(binary_entries, columns=['regulator', 'target', 'value'])
        binary_matrix = binary_df.pivot(index='regulator', columns='target', values='value').fillna(0).astype(int)

        # Step 3: Save the wide-form matrix to CSV
        binary_matrix.to_csv(f'{output_dir}/{MJ}_binary_matrix.csv')

        print(f"Processed {MJ}: binary_matrix saved with shape {binary_matrix.shape}.")
