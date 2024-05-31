from tgnn import *
import matplotlib.pyplot as plt
import random
from torch.nn import CosineSimilarity
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from scipy.stats import iqr

import argparse
import os

import seaborn as sns
from sklearn.metrics.cluster import normalized_mutual_info_score

def load_dataset(folder, feature):
    snapshots = []
    for i in range(4):
        snap = Data()
        snap.x = torch.load(f'{folder}/snapshot_{i}_x_{feature}.pt')
        snap.edge_index = torch.load(f'{folder}/snapshot_{i}_edge_index.pt')
        snapshots.append(snap)
    return snapshots

def cosine_similarity(node_states, hop, feature, textonly):
    cosine = CosineSimilarity(dim=1, eps=1e-6)

    label = {
        1: 'before',
        2: 'during',
        3: 'after'
    }

    plt.figure(figsize=(16,9))
    for i in range(1,4):
        ti = node_states[i][hop]
        sim = cosine(node_states[0][hop], ti)
        plt.hist(sim.detach().numpy(), label=f'{label[i]}', bins=30, alpha=0.6)
        plt.legend(loc='upper right', fontsize=30)
        plt.yscale('log')
        plt.xlabel('Cosine similarity', fontsize=30)
        plt.ylabel('Frequency', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        #plt.title(f'Similarity between structural node embeddings over time ({hop+1}-hop)')
    if textonly:
        plt.savefig(f'results/cosine_sbert_only.pdf', bbox_inches='tight')
    else:
        plt.savefig(f'results/cosine_{feature}_{hop+1}.pdf', bbox_inches='tight')
    plt.clf()
    print("Cosine similarity distributions saved")
    #plt.show()
    
def create_dir():
    if not os.path.exists('results'):
        os.makedirs(args.directory)
        print(f"Directory results created.")

def main():
    parser = argparse.ArgumentParser()
    # Define the arguments
    parser.add_argument('--data', type=str, required=True, help='Dataset folder as a string')
    parser.add_argument('--feature', type=str, choices=['struct', 'text'], required=True, help='Initial node features: structural or textual')
    parser.add_argument('--sbert_only', action='store_true', help='Boolean flag, if set, only SBERT embedding is considered')
    
    # Parse the arguments
    args = parser.parse_args()
    
    print('Loading dataset...')
    snapshots = load_dataset(args.data, "text" if args.sbert_only else args.feature)
    
    print(f'Experiments with {args.data} using {args.feature} feature') 
    
    create_dir()
    
    if args.sbert_only:
        hop = 0
        node_states = {k:{} for k in range(len(snapshots))}
        for i in range(len(snapshots)):
            node_states[i][0] = snapshots[i].x
        cosine_similarity(node_states, hop, "", args.sbert_only)
        print('End')
    else:
        device = torch.device('cuda')
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        np.random.seed(1)
        random.seed(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
        
        hidden_conv1 = 64
        hidden_conv2 = 32
        
        print('Training the model...')
        model, node_states = train_roland(snapshots, hidden_conv1, hidden_conv2)
        print('Training finished')
        
        print('Computing the cosine similarities...')
        for i in [0,1]:
            cosine_similarity(node_states, i, args.feature, args.sbert_only)
              
              
        print('Computing clusters and NMIs...')
        hop=0
        n_clusters=8
              
        labels = []
        for i in range(0,4):
            X = node_states[i][hop].detach().numpy()
            brc = Birch(n_clusters=n_clusters) #default parameters
            brc.fit(X)
            y=brc.predict(X)
            #print(silhouette_score(X,y))
            labels.append(y)
              
        group = ['Before','During','After','After2']

        nmis = [[round(normalized_mutual_info_score(labels[j], labels[i]),1) for i in range(0,4)] for j in range(0,4)]

        plt.figure(figsize=(16,9))

        sns.heatmap(nmis, cmap='Blues', annot=True, xticklabels=group, yticklabels=group,\
                    annot_kws={'fontsize': 30})
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        #plt.title(f'NMI between structural user clustering over time ({hop+1}-hop)')
        plt.savefig(f'results/NMI_{args.feature}_{hop+1}.pdf', bbox_inches='tight')
        #plt.show()
        plt.clf()
        print('NMI heatmaps saved')
        
        print('Computing IQR...')
        label = {
            0: 'before',
            1: 'during',
            2: 'after',
            3: 'after2'
        }

        plt.figure(figsize=(16,9))
        for i in range(0,4):
            ti = node_states[i][hop].detach().numpy()
            g = iqr(ti,axis=1)
            plt.hist(g, label=f'{label[i]}', bins=30, alpha=0.6)
            plt.legend(loc='upper right', fontsize=30)
            plt.yscale('log')
            plt.xlabel('Iqr', fontsize=30)
            plt.ylabel('Frequency', fontsize=30)
            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            #plt.title(f'Iqr of node embeddings with text ({hop+1}-hop)')
    
        plt.savefig(f'results/iqr_{args.feature}_{hop+1}.pdf', bbox_inches='tight')
        plt.clf()
        #plt.show()
        print('IQRs saved')
        print('End')
              
if __name__=='__main__':
    main()