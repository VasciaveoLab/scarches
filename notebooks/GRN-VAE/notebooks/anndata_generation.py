import scanpy as sc
import scarches as sca
import numpy as np
import os
import matplotlib.pyplot as plt

n_top_genes = [2000, 2500, 3000, 4000, None]
percs = [1, 2, 5, 10, 25, 50]

# Path to the folder containing the files
data_path = "/shares/vasciaveo_lab/data/nepc_organoid_project/new_data/MJ005/filtered_feature_bc_matrix"

# Load the 10X Genomics formatted data
adata = sc.read_10x_mtx(
    data_path,              # Directory with the matrix.mtx, barcodes.tsv, genes.tsv files
    var_names='gene_symbols',  # Use gene symbols as variable names
    cache=True                 # Cache the result for faster loading next time
)

adata.layers['counts'] = adata.X.copy()

adata_marker_path = '/home/lgolinelli/git/GRN-VAE/expimap_outputs/adata_markers.h5ad'
adata_marker = sc.read_h5ad(adata_marker_path)

if (adata_marker.obs_names == adata.obs_names).all():
    clusters_path = '/home/lgolinelli/git/GRN-VAE/outputs/acdc_pax_3_clusters.npy'
    adata.obs['cancer_subtype'] = adata_marker.obs['cancer_subtype'].values.copy()
else:
    raise ValueError("Observation names do not match between adata and adata_marker.")

del adata_marker

for perc in percs:
    adata_perc = adata.copy()
    annotation_path = f'/home/lgolinelli/git/scarches-1/notebooks/GRN-VAE/binary_matrices/tsv_outputs/regulator_gene_{perc}percent.tsv'
    sca.utils.add_annotations(adata_perc, annotation_path, min_genes=1, clean=False, genes_use_upper=False)

    adata_perc._inplace_subset_var(adata_perc.varm['I'].sum(1)>0)

    sc.pp.normalize_total(adata_perc)
    sc.pp.log1p(adata_perc)
    adata_perc.layers['X_normalized'] = adata_perc.X.copy()

    for n_top_gene in n_top_genes:
        adata_top = adata_perc.copy()
        subset = True if n_top_gene is not None else False
        sc.pp.highly_variable_genes(
            adata_top,
            n_top_genes=n_top_gene,
            subset=subset)

        adata_top.X = adata_top.layers["counts"].copy()
        # Filter out any annotations (terms) with less than 12 genes.
        select_terms = adata_top.varm['I'].sum(0)>1
        adata_top.uns['terms'] = np.array(adata_top.uns['terms'])[select_terms].tolist()
        adata_top.varm['I'] = adata_top.varm['I'][:, select_terms]

        # Filter out genes that are not annotated to any term after HVG selection
        adata_top._inplace_subset_var(adata_top.varm['I'].sum(1)>0)

        pic_path = f'/home/lgolinelli/git/scarches-1/notebooks/GRN-VAE/anndata_inputs/pics/percent_{perc}_top_genes_{n_top_gene}/'
        os.makedirs(pic_path, exist_ok=True)
        for layer in ['X_normalized', 'counts']:
            adata_top.X = adata_top.layers[layer].copy()
            sc.pp.pca(adata_top)
            sc.pp.neighbors(adata_top)
            sc.tl.umap(adata_top)
            sc.pl.umap(adata_top, color='cancer_subtype', title=layer, show=False)
            out_file = os.path.join(pic_path, f'{layer}_cancer_subtype.png')
            plt.savefig(out_file, bbox_inches='tight')
            plt.close()

        adata_top.var['gene_idx'] = np.arange(adata_top.shape[1])
        adata_top.var['counts_in_terms'] = 0
        adata_top.var['gene_as_regulator'] = False
        for gene_name in adata_top.var_names:
            gene_index = adata_top[:, gene_name].var['gene_idx']
            gene_count_in_term = adata_top.varm['I'][gene_index].sum()
            adata_top.var.loc[gene_name, 'counts_in_terms'] = gene_count_in_term
            adata_top.var.loc[gene_name, 'gene_as_regulator'] = gene_name in adata_top.uns['terms']
        
        # Check if 'Nsd2' is present as target and/or regulator
        nsd2_as_target = 'Nsd2' in list(adata_top.var_names)
        nsd2_as_regulator = 'Nsd2' in adata_top.uns['terms']

        adata_top.uns['nsd2_as_target'] = nsd2_as_target
        adata_top.uns['nsd2_as_regulator'] = nsd2_as_regulator

        # Save the processed AnnData object
        adata_top_path = f'/home/lgolinelli/git/scarches-1/notebooks/GRN-VAE/anndata_inputs/adata_top_{perc}_top_genes_{n_top_gene}.h5ad'
        os.makedirs(os.path.dirname(adata_top_path), exist_ok=True)
        adata_top.write(adata_top_path)

        # Write results to a single TSV file (append mode)
        nsd2_notes_path = f'/home/lgolinelli/git/scarches-1/notebooks/GRN-VAE/anndata_inputs/nsd2_status.tsv'
        with open(nsd2_notes_path, 'a') as f:
            f.write("perc\tn_top_genes\tNsd2_as_target\tNsd2_as_regulator\n")
            f.write(f"{perc}\t{n_top_gene}\t{int(nsd2_as_target)}\t{int(nsd2_as_regulator)}\n")

        print(f"Processed dataset with perc {perc} and top genes {n_top_gene}.")