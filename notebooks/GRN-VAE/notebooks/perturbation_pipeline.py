import warnings
warnings.simplefilter(action='ignore')

import scanpy as sc
import numpy as np
import pandas as pd
import torch
import scarches as sca
import sys

sc.set_figure_params(frameon=False)
sc.set_figure_params(dpi=200)
sc.set_figure_params(figsize=(4, 4))
torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)

perc = 1
top_genes = None
condition_key = 'cancer_subtype'

adata_path = f'/home/lgolinelli/git/scarches-1/notebooks/GRN-VAE/anndata_inputs/adata_top_{perc}_top_genes_{top_genes}.h5ad'
model_path = f'/home/lgolinelli/git/scarches-1/notebooks/GRN-VAE/trained_models/model_{perc}_top_genes_{top_genes}_{condition_key}'

adata = sc.read_h5ad(adata_path)

#if condition_key != 'placeholder_condition':
#    adata.obs["placeholder_condition"] = "placeholder"

vae = sca.models.EXPIMAP.load(model_path, adata, map_location='cpu')

full_model = vae.model
encoder = full_model.encoder
decoder = full_model.decoder

adata = sc.read_h5ad(adata_path)

pert_gene_adata = vae.perturb_genes(
    adata=adata,
    genes='Nsd2',
    perturb_type='KO',
    group_key='cancer_subtype',
    category='CRPC-NE',
    obs_key='perturbation',
    condition_key='cancer_subtype',
    new_condition=None
)


adata.write_h5ad('/home/lgolinelli/git/scarches-1/notebooks/GRN-VAE/notebooks/perturbation_adata.h5ad')