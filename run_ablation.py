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

from scipy.stats import wasserstein_distance

def load_dataset(folder, feature):
    snapshots = []
    for i in range(4):
        snap = Data()
        snap.x = torch.load(f'{folder}/snapshot_{i}_x_{feature}.pt')
        snap.edge_index = torch.load(f'{folder}/snapshot_{i}_edge_index.pt')
        snapshots.append(snap)
    return snapshots
    
def create_dir():
    if not os.path.exists('results'):
        os.makedirs(args.directory)
        print(f"Directory results created.")

def main():
    parser = argparse.ArgumentParser()
    # Define the arguments
    parser.add_argument('--data', type=str, required=True, help='Dataset folder as a string')
    # Parse the arguments
    args = parser.parse_args()
    
    print('Loading dataset...')
    snapshots = load_dataset(args.data, "text")
    
    print(f'Experiments with {args.data} using text feature') 
    
    create_dir()
    
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
    ablation_str = ['Temporal Embeddings', 'w/o UPDATE','w/o GNN', 'w/o current-features', 'w/o current-topology']

    cosine = CosineSimilarity(dim=1, eps=1e-6)

    cosines = {}

    for ablation in ablation_str:
        print(f'Training the model {ablation}')
        if ablation == 'Temporal Embeddings':
            model, node_states = train_roland(snapshots, hidden_conv1, hidden_conv2)
        else:
            model, node_states = train_roland_ablation(snapshots, hidden_conv1, hidden_conv2, ablation=ablation)
        print('Training finished')

        cosine_sim_1 = cosine(node_states[0][0], node_states[2][0])
        cosine_sim_2 = cosine(node_states[0][0], node_states[1][0])
        # Compute Wasserstein Distance
        w_dist = wasserstein_distance(cosine_sim_1.detach().numpy(), cosine_sim_2.detach().numpy())
        cosines[ablation] = w_dist

    print('Ablation end')

    print("EMD Distances")
    for k,v in cosines.items():
        print(f'{k}: {v}')
    print('End')
    """
    # Convert dictionary to DataFrame
    cosine_data = [(label, value) for label, values in cosines.items() for value in values]
    df = pd.DataFrame(cosine_data, columns=["Label", "Value"])
        
    # Create the violin plot
    plt.figure(figsize=(16,9))
    sns.violinplot(x="Label", y="Value", data=df)
    plt.legend(loc='upper right', fontsize=30)
    plt.xlabel('Embedding', fontsize=30)
    plt.ylabel('Cosine similarity', fontsize=30)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=30)
    plt.savefig(f'results/ablation.pdf', bbox_inches='tight')    

    print('Violion plots saved')
    """
if __name__=='__main__':
    main()