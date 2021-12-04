import torch	
from transformers import BertTokenizer, BertForMaskedLM	


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained('bert-base-uncased') 
model.cuda()


#tokenizer.save_vocabulary("./","rorberta-large")
#print(tokenizer._convert_id_to_token(11522))
#tokenizer.save_vocabulary("./","berta-base-cased")
text="A logical corollary is that inflation cannot be triggered by increasing wages, farm prices, or health care costs."
#print(tokenizer.convert_ids_to_tokens(tokenizer(text)["input_ids"]))
#print(list(tokenizer.vocab.keys())[100:120])
