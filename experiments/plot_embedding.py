import matplotlib.pyplot as plt
import dill as pickle
import numpy as np
from test_bertformasklm import FillwithBert
from sklearn.decomposition import PCA

def main():
    model=FillwithBert("bert-base-cased")
    embed_weight=model.plm.get_input_embeddings().weight.detach().numpy()
    embed_afterpca=PCA(n_components=3).fit(embed_weight).transform(embed_weight)
    with open("./others/bert_base_cased_embed_pca3.pkl","wb") as f:
        pickle.dump(embed_afterpca,f)
    x=embed_afterpca[:,0]
    y=embed_afterpca[:,1]
    z=embed_afterpca[:,2]
    fig= plt.figure(1,figsize=(5,5))
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(x,y,z,s=1,cmap="RdYlGn")
    plt.savefig("./imgs/bert_base_cased_embed_pca3.svg",dpi=900,format='svg')

if __name__ == '__main__':
    model=FillwithBert("bert-base-cased")
    embed_weight=model.plm.get_input_embeddings().weight.detach().numpy()
    u,sigma,v=np.linalg.svd(np.ascontiguousarray(embed_weight))
    print(sigma)