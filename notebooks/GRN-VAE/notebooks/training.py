import scanpy as sc
import scarches as sca
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


n_top_genes = [2000, 2500, 3000, 4000, None]
percs = [1, 2, 5, 10, 25, 50]
conditions = [True, False]

adatas_path = '/home/lgolinelli/git/scarches-1/notebooks/GRN-VAE/anndata_inputs'
models_path = '/home/lgolinelli/git/scarches-1/notebooks/GRN-VAE/trained_models'

for condition in conditions:
    for perc in percs:
        for n_top_gene in n_top_genes:
                adata_path = os.path.join(adatas_path, f'adata_top_{perc}_top_genes_{n_top_gene}.h5ad')
                adata = sc.read(adata_path)

                adata.obs['placeholder_condition'] = 'placeholder'  # Placeholder for condition key
                condition = 'cancer_subtype' if condition else 'placeholder_condition'

                model = sca.models.EXPIMAP(
                adata=adata,
                condition_key=condition,
                hidden_layer_sizes=[300, 300, 300],
                recon_loss='nb',
                soft_mask=False,
                )

                early_stopping_kwargs = {
                    "early_stopping_metric": "val_unweighted_loss",
                    "threshold": 0,
                    "patience": 50,
                    "reduce_lr": True,
                    "lr_patience": 13,
                    "lr_factor": 0.1,
                }

                model.train(
                    n_epochs=4,
                    alpha_epoch_anneal=2,
                    alpha=1.0,
                    alpha_kl=0.5,
                    weight_decay=0.,
                    early_stopping_kwargs=early_stopping_kwargs,
                    use_early_stopping=True,
                    seed=2020
                )
                model_name = f'intr_cvae_{perc}_top_genes_{n_top_gene}_{condition}'
                model_path = os.path.join(models_path, model_name)
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                model.save(model_path, overwrite=True, save_anndata=False)
