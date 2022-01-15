import argparse
parser = argparse.ArgumentParser(description="Training scSemiAE")
# Training settings:
parser.add_argument("-dpath", "--data_path", default="./dataset/", help="path to the dataset folder")
parser.add_argument("-spath", "--save_path", default="./output/", help="path to output directory")

parser.add_argument("-lsize", "--lab_size", type=int, default=10, help="labeled set size for each cell type (default: 10)")
parser.add_argument("-lratio", "--lab_ratio", type=float, default=-1, help="labeled set ratio for each cell type (default: -1)")

parser.add_argument("-s", "--seed", type=int, default=0, help="random seed for loading dataset (default: 0)")
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('-pretrain_batch', '--pretrain_batch', type=int, help="Batch size for pretraining.Default:100", default=100)
parser.add_argument('-nepoch', '--epochs', type=int, help='number of epochs to train for', default=60)
parser.add_argument('-nepoch_pretrain', '--epochs_pretrain', type=int, help='number of epochs to pretrain for', default=50)
parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate for the model, default=0.001', default=0.001)
parser.add_argument('-lrS', '--lr_scheduler_step', type=int, help='StepLR learning rate scheduler step, default=10', default=10)
parser.add_argument('-lrG', '--lr_scheduler_gamma', type=float, help='StepLR learning rate scheduler gamma, default=0.5', default=0.5)
parser.add_argument('-lbd', '--Lambda', type=float, help='weight for L2, default=1', default=1)
parser.add_argument('-v', '--visual', type=bool, help='visualization of data. default=False', default=False)
args = parser.parse_args()

# make sure saving path exists
import os
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

import torch
from model.scSemiAE import scSemiAE
from model.dataset import ExperimentDataset
from model.inference import knn_infer,  louvain_cluster_infer, kmeans_cluster_infer
from model.metrics import compute_scores, compute_scores_for_cls
import scanpy as sc
import anndata
import torch
import matplotlib.pyplot as plt
from data import Data
import warnings
warnings.filterwarnings("ignore")

def main():

    # Set device
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'
    args.device = device

    # Data loading
    if args.lab_ratio != -1:  # use labeled ratio
        dataset = Data(args.data_path, labeled_ratio=args.lab_ratio, seed=args.seed)
    else:  # use labeled size
        dataset = Data(args.data_path, labeled_size=args.lab_size, seed=args.seed)
    data, lab_full, labeled_idx, unlabeled_idx, info = dataset.load_all()

    # print("############################# scSemiAE")
    data = torch.tensor(data, dtype=torch.float)

    cell_id = info["cell_id"]
    labeled_data = data[labeled_idx, :]
    labeled_lab = lab_full[labeled_idx].tolist()
    unlabeled_lab = lab_full[unlabeled_idx].tolist()
    labeled_cellid = cell_id[labeled_idx].tolist()

    pretrain_data = ExperimentDataset(data, cell_id, lab_full)
    labeled_data = ExperimentDataset(labeled_data, labeled_cellid, labeled_lab)

    model = scSemiAE(args, labeled_data, pretrain_data, hid_dim_1=500, hid_dim_2=50)
    embeddings = model.train()

    if args.save_path:
        embd_save_path = args.save_path + "scSemiAE_embeddings.csv"
        embeddings.to_csv(embd_save_path)

    embeddings = embeddings.values

    # knn infer
    unlabeled_lab_knn_pred = knn_infer(embeddings, lab_full, labeled_idx, unlabeled_idx)
    scores_knn = compute_scores(unlabeled_lab, unlabeled_lab_knn_pred)
    print("KNN:")
    print(scores_knn)

    # cluster
    pred = louvain_cluster_infer(embeddings, unlabeled_idx)
    scores_louvain = compute_scores_for_cls(unlabeled_lab, pred)
    print("Louvain:")
    print(scores_louvain)

    # cluster
    pred_kmcls = kmeans_cluster_infer(embeddings, unlabeled_idx, n_cls=len(set(info["cell_label"])))
    scores_kmcls = compute_scores_for_cls(unlabeled_lab, pred_kmcls)
    print("kmCLS:")
    print(scores_kmcls)
    print("######################## over!")

    if args.visual:
        em = anndata.AnnData(embeddings[unlabeled_idx])
        em.obs['cell_label'] = [info['cell_label'][i] for i in lab_full[unlabeled_idx]]
        #em.obs['batch'] = [info['batch_name'][i] for i in info['batch'][unlabeled_idx]]
        sc.pp.neighbors(em, n_neighbors=30, use_rep='X')
        # sc.tl.louvain(em)
        # em.obs['kmeans'] = pred_kmcls
        sc.tl.umap(em)
        sc.pl.umap(em, color=['cell_label'])
        plt.savefig("cls.png")
        plt.show()

if __name__ == "__main__":
    main()

