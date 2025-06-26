#!/usr/bin/env python
import os
import argparse

import scanpy as sc
import scarches as sca

def parse_args():
    p = argparse.ArgumentParser(description="Train one EXPIMAP model")
    p.add_argument("--condition-flag",  type=int, choices=[0,1], required=True,
                   help="0 → placeholder_condition, 1 → cancer_subtype")
    p.add_argument("--perc",            type=int, required=True,
                   help="percent value, e.g. 1,2,5,10,25,50")
    p.add_argument("--n-top-genes",     type=str, required=True,
                   help="2000,2500,3000,4000 or 'None'")
    p.add_argument("--adatas-path",     type=str, required=True,
                   help="path to your anndata inputs folder")
    p.add_argument("--models-path",     type=str, required=True,
                   help="path to your trained_models output folder")
    return p.parse_args()

def main():
    args = parse_args()

    # interpret flags
    cond_key = "cancer_subtype" if args.condition_flag==1 else "placeholder_condition"
    n_top   = None if args.n_top_genes=="None" else int(args.n_top_genes)

    # load
    inpath = os.path.join(
        args.adatas_path,
        f"adata_top_{args.perc}_top_genes_{n_top}.h5ad"
    )
    
    adata = sc.read(inpath)
    adata.obs["placeholder_condition"] = "placeholder"

    # build & train
    model = sca.models.EXPIMAP(
        adata=adata,
        condition_key=cond_key,
        hidden_layer_sizes=[300,300,300],
        recon_loss="nb",
        soft_mask=False,
    )

    es_kwargs = {
        "early_stopping_metric": "val_unweighted_loss",
        "threshold": 0,
        "patience": 50,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }

    model.train(
        n_epochs=400,
        alpha_epoch_anneal=100,
        alpha=1.0,
        alpha_kl=0.5,
        weight_decay=0.0,
        early_stopping_kwargs=es_kwargs,
        use_early_stopping=True,
        seed=2020,
    )

    # save
    outdir = os.path.join(
        args.models_path,
        f"model_{args.perc}_top_genes_{n_top}_{cond_key}"
    )
    os.makedirs(outdir, exist_ok=True)
    model.save(outdir, overwrite=True, save_anndata=False)

if __name__=="__main__":
    main()
