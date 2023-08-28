try:
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
except:
    pass

import psutil
from torch.utils.tensorboard import SummaryWriter
from arg_parser import parser
import torch
import os
import anndata as ad
import scanpy as sc
import anndata
import pandas as pd
import logging
from pathlib import Path
from scNTImpute.models import scNTImpute

from scNTImpute.trainers import UnsupervisedTrainer
from scNTImpute.eval_utils import evaluate
import matplotlib
from scNTImpute.models.BatchClassifier import BatchClassifier
from scNTImpute.trainers.BatchAdversarialTrainer import BatchAdversarialTrainer
from scNTImpute.trainers.MMDTrainer import MMDTrainer

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    args = parser.parse_args()

    matplotlib.use('Agg')
    sc.settings.set_figure_params(
        dpi=args.dpi_show, dpi_save=args.dpi_save, facecolor='white', fontsize=args.fontsize, figsize=args.figsize)

    # load dataset
    mp_csv = ['human_islet.csv']
    mp_adatas = []
    for fpath in mp_csv:
        df = pd.read_csv(fpath, index_col=0)
        adata_1 = ad.AnnData(X=df.iloc[:, 1:], obs=df.iloc[:, :1])#1   2
        mp_adatas.append(adata_1)
    adata = ad.concat(mp_adatas, label="batch_indices")
    adata.obs_names_make_unique()
    adata.obs['total_counts'] = adata.X.sum(1)

    if hasattr(args, 'pathway_csv_path') and args.pathway_csv_path:
        mat = pd.read_csv(args.pathway_csv_path, index_col=0)
        if args.trainable_gene_emb_dim == 0:
            genes = sorted(list(set(mat.index).intersection(adata.var_names)))
            logger.info(f'Using {mat.shape[1]}-dim fixed gene embedding for {len(genes)} genes appeared in both the gene-pathway matrix and the dataset.')
            adata = adata[:, genes]
            adata.varm['gene_emb'] = mat.loc[genes, :].values
        else:
            logger.info(f'{mat.shape[1]} dimensions of the gene embeddings will be trainable. Keeping all genes in the dataset') 
            mat = mat.reindex(index=adata.var_names, fill_value=0.0)
            adata.varm['gene_emb'] = mat.values

    start_mem = psutil.Process().memory_info().rss
    logger.info(f'Before model instantiation and training: {psutil.Process().memory_info()}')

    if args.model.startswith('scNTImpute'):
        model = scNTImpute(
            n_trainable_genes = adata.n_vars,
            n_trainable_cells=adata.n_obs,
            adata_scNTImpute=adata,
            dthre=args.dthre,
            zi_hidden_size=(256,),
            norm_genes=True,
            batch_size=adata.n_obs,
            loglik_initial_value=torch.zeros([adata.n_vars]),
            drop_process=False,
            n_epoch=args.n_epochs,
            n_batches = adata.obs.batch_indices.nunique(),
            n_topics = args.n_topics,
            trainable_gene_emb_dim = args.trainable_gene_emb_dim,
            hidden_sizes = args.hidden_sizes,
            bn = not args.no_bn,
            dropout_prob = args.dropout_prob,
            norm_cells = args.norm_cells,
            normed_loss = args.normed_loss,
            enable_batch_bias = args.batch_bias,
            rho_fixed_emb=adata.varm['gene_emb'].T if 'gene_emb' in adata.varm else None,
            device = torch.device(args.device)
        )
    else:
        raise NotImplementedError

    if args.model.endswith('batch'):
        args.model = f'{args.model}{args.adv_loss}'
        trainer = BatchAdversarialTrainer(
            model,
            BatchClassifier(model.n_topics, model.n_batches, (model.n_topics, model.n_topics // 2), device = torch.device(args.device), adversarial_loss=args.adv_loss),
            adata,
            train_instance_name = f"{args.dataset_str}_{args.model}{args.log_str}_seed{args.seed}",
            seed = args.seed,
            ckpt_dir = args.ckpt_dir,
            batch_size = args.batch_size,
            test_ratio = args.test_ratio,
            data_split_seed = args.data_split_seed,
            restore_epoch = args.restore_epoch,
            init_lr = args.lr,
            lr_decay = args.lr_decay,
            batch_clf_init_lr = args.batch_clf_lr,
            batch_clf_lr_decay = args.batch_clf_lr_decay
        )
    elif args.model.endswith('mmd'):
        trainer = MMDTrainer(
            model,
            adata,
            train_instance_name=f"{args.dataset_str}_{args.model}{args.log_str}_seed{args.seed}",
            seed=args.seed,
            ckpt_dir=args.ckpt_dir,
            batch_size=args.batch_size,
            test_ratio=args.test_ratio,
            data_split_seed=args.data_split_seed,
            restore_epoch=args.restore_epoch,
            init_lr=args.lr,
            lr_decay=args.lr_decay
        )
    else:
        if not args.batch_bias:
            args.model = f'{args.model}nobatch'
        trainer = UnsupervisedTrainer(
            model,
            adata,
            train_instance_name=f"{args.dataset_str}_{args.model}{args.log_str}_seed{args.seed}",
            seed=args.seed,
            ckpt_dir=args.ckpt_dir,
            batch_size=adata.n_obs,
            test_ratio=args.test_ratio,
            data_split_seed=args.data_split_seed,
            restore_epoch=args.restore_epoch,
            init_lr=args.lr,
            lr_decay=args.lr_decay
        )

    writer = SummaryWriter(os.path.join(trainer.ckpt_dir, 'tensorboard'))

    trainer.train(
        n_epochs = args.n_epochs,
        eval_every = args.eval_every,
        n_samplers=args.n_samplers,
        kl_warmup_ratio=args.kl_warmup_ratio,
        min_kl_weight=args.min_kl_weight,
        max_kl_weight=args.max_kl_weight,
        save_model_ckpt=not args.no_model_ckpt,
        eval=not args.no_eval,
        record_log_path = os.path.join(trainer.ckpt_dir, 'record.tsv'),
        writer = writer,
        eval_result_log_path = os.path.join(args.ckpt_dir, 'result.tsv'),
        eval_kwargs=dict(resolutions=args.resolutions, cell_type_col='cell_type'),
        clf_cutoff_ratio = args.clf_cutoff_ratio,
        clf_warmup_ratio = args.clf_warmup_ratio,
        min_clf_weight = args.min_clf_weight,
        max_clf_weight = args.max_clf_weight,
        g_steps = args.g_steps,
        d_steps = args.d_steps,
        mmd_warmup_ratio = args.mmd_warmup_ratio,
        min_mmd_weight = args.min_mmd_weight,
        max_mmd_weight = args.max_mmd_weight,
    )

    mem_cost = psutil.Process().memory_info().rss - start_mem
    logger.info(f'After model instantiation and training: {psutil.Process().memory_info()}')
    train_instance_name, clustering_input, ckpt_dir = trainer.train_instance_name, trainer.model.clustering_input, trainer.ckpt_dir

    if args.target_h5ad_path:
        target_adata = anndata.read_h5ad(args.target_h5ad_path)
        assert adata.n_vars == target_adata.n_vars
        args.target_dataset_str = Path(args.target_h5ad_path).stem
        del adata, trainer
    else:
        target_adata = adata

    if 'delta' not in target_adata.obsm:
        model.get_cell_embeddings_and_nll(target_adata, emb_names='delta')
    emb = anndata.AnnData(X = target_adata.obsm['delta'], obs = target_adata.obs)
    emb.write_h5ad(os.path.join(ckpt_dir, f"{args.dataset_str}_{args.model}_seed{args.seed}.h5ad"))
    result = evaluate(target_adata,
        embedding_key = model.clustering_input,
        resolutions = args.resolutions,
        plot_fname = f'{train_instance_name}_{clustering_input}_eval',
        plot_dir = ckpt_dir,
        writer = writer,
        color_by=args.color_by,
        umap_kwargs=dict(size=args.point_size if args.point_size else None)
    )
    if args.target_h5ad_path:
        if args.restore_epoch:
            log_path = os.path.join(args.ckpt_dir, '..', 'transfer.tsv')
        else:
            log_path = os.path.join(args.ckpt_dir, 'transfer.tsv')
        with open(log_path, 'a+') as f:
            f.write(f'{args.dataset_str}\t{args.target_dataset_str}\t{args.model}{args.log_str}\t{args.seed}\t{result["ari"]}\t{result["nmi"]}\t{result["asw"]}\t{result["ebm"]}\t{result["k_bet"]}\t{mem_cost}\n')
    else:
        if args.restore_epoch:
            log_path = os.path.join(args.ckpt_dir, '..', 'table1.tsv')
        else:
            log_path = os.path.join(args.ckpt_dir, 'table1.tsv')
        with open(log_path, 'a+') as f:
            f.write(f'{args.dataset_str}\t{args.model}{args.log_str}\t{args.seed}\t{result["ari"]}\t{result["nmi"]}\t{result["asw"]}\t{result["ebm"]}\t{result["k_bet"]}\t{mem_cost/1024}\n')
