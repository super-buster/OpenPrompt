import matplotlib.pyplot as plt
import dill as pickle


with open("bert_base_uncased_embed_pca2.pkl","rb") as f:
    M=pickle.load(f)

x=M[:,0]
y=M[:,1]

plt.scatter(x,y)
plt.savefig("./bert_base_uncased_embed_pca2.png")