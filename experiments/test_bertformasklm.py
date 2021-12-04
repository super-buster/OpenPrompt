import torch
import numpy as np
import re
import dill as pickle
from sklearn.decomposition import PCA

from transformers.utils.dummy_pt_objects import BertForMaskedLM

class FillwithBert(object):
  def __init__(self,plm_name:str) -> None:
      super().__init__()
      if plm_name.startswith("bert"):
        from transformers import BertTokenizer, BertForMaskedLM
        self.tokenizer=BertTokenizer.from_pretrained(plm_name)
        self.plm= BertForMaskedLM.from_pretrained(plm_name).eval()
      elif plm_name.startswith("roberta"):
        from transformers import RobertaTokenizer, RobertaForMaskedLM
        self.tokenizer=RobertaTokenizer.from_pretrained(plm_name)
        self.plm= RobertaForMaskedLM.from_pretrained(plm_name).eval()

  def predict(self,input: str):
      input = self.tokenizer(input,return_tensors="pt")
      position=[]
      for index,item in enumerate(list(input['input_ids'][0])):
        if item == self.tokenizer.mask_token_id:
          position.append(index)
      with torch.no_grad():
        outputs = self.plm(**input)
      return outputs[0][0].detach().numpy(),position


def main():
  model=FillwithBert("bert-base-uncased")
  embed_weight=model.plm.get_input_embeddings().weight.detach().numpy()
  embed_afterpca=PCA(n_components=2).fit_transform(embed_weight)
  with open("bert_base_uncased_embed_pca2.pkl","wb") as f:
    pickle.dump(embed_afterpca,f)
  detect_words=["good","bad","awful","wonderful","fantastic","odd","terrible","well","fine","poor","lousy","disgusting","shit","the"]
  detect_words_id=model.tokenizer.convert_tokens_to_ids(detect_words)
  words_prob={} 
  input="I hate coffee, because it tastes [MASK] [MASK] ."
  pred_scores,position=model.predict(input)
  for id,item in enumerate(detect_words_id): 
    words_prob[detect_words[id]]=pred_scores[-3][item]
  print(words_prob)
  filled_words=[model.tokenizer._convert_id_to_token(np.argmax(pred_scores[p])) for p in position]
  print("the most possible word filled in the mask place is {}".format(filled_words))

main()