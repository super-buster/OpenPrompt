import argparse,os
import dill
from tqdm import tqdm
from logging import log, raiseExceptions
from collections import namedtuple
import torch
import numpy as np
from torch._C import device
from openprompt.data_utils import load_dataset
from openprompt.config import get_user_config
from openprompt.plms import load_plm, load_plm_from_config
from openprompt.utils.cuda import model_to_device
from openprompt.prompts import load_template, load_verbalizer
from openprompt.pipeline_base import PromptForClassification
from openprompt import PromptDataLoader
from openprompt.utils.metrics import classification_metrics

'''
creator: yzx
datetime: 2021-11-9
function: Do ensemble for heterogeneous prompts
How to use: python template_ensemble.py --E agnews_roberta-large_xxx --metrics accuracy micro-f1 --seed 123
'''

_CONSTANT={
    "num_test_sample": 32,
}

PLM=namedtuple('PLM',['plm','plm_tokenizer','plm_config','plm_wrapper_class'])
PROMPT_MODLE=namedtuple('PROMPT_MODEL',['model','data_loader'])

def check_gpu(args):
    devices= os.environ.get('cuda_visible_diveces') if os.environ.get('cuda_visible_diveces') is not None else os.environ.get('CUDA_VISIBLE_DIVECES')
    assert devices is not None,"you must set cuda_visible_diveces !"
    assert len(devices.split(','))>= len(args.ensemble)

def build_dataloader(dataset, template, tokenizer,tokenizer_wrapper_class, config, split):
    dataloader = PromptDataLoader(
        dataset = dataset, 
        template = template, 
        tokenizer = tokenizer, 
        tokenizer_wrapper_class=tokenizer_wrapper_class, 
        batch_size = config[split].batch_size,
        shuffle = config[split].shuffle_data,
        teacher_forcing = config[split].teacher_forcing if hasattr(config[split],'teacher_forcing') else None,
        predict_eos_token = True if config.task == "generation" else False,
        **config.dataloader
    )
    return dataloader

def get_args():
    parser=argparse.ArgumentParser(prog='template ensemble',formatter_class=argparse.ArgumentDefaultsHelpFormatter,prefix_chars="-",description="run some prompt models to do ensemble")
    parser.add_argument('--ensemble','--E',metavar="ENSEMBLE",nargs="+",type=str,help='ensemble models root path')
    parser.add_argument('--metrics','--M',metavar='METRICS',nargs="+",type=str,default="accuracy",choices=["accuracy","micro-f1","macro-f1"],help="choose which metrics to use")
    parser.add_argument('--seed','--S',metavar='SEED',nargs="+",type=str,default="123",help="choose which seed model to use")
    args=parser.parse_args()
    return args,parser

def get_ensemble_cfgs(args):
    logs_path=["../experiments/logs/"+ p + "/config.yaml" for p in args.ensemble]
    configs=[]  
    for item in logs_path:
        configs.append(get_user_config(item))
    return configs

def multi_forward(prompt_models:PROMPT_MODLE) -> list:
    logits,labels=[],[]
    for order in range(len(prompt_models)):
        outputs=[]
        for step,input in enumerate(tqdm(prompt_models[order].data_loader)):
            input=input.to('cuda:{}'.format(order))
            if order==0:
                labels.extend(input['label'].cpu().tolist())
            output=prompt_models[order].model(input)
            outputs.extend(output.cpu().tolist())
        logits.append(outputs)
    return logits,labels

def ensemble(args,cfgs,method,logits:list,lables:list):
    for metric in args.metrics:
        # make sure the performance is equal to origin test method
        for order,logit in enumerate(logits):
            name=cfgs[order].logging.unique_string
            pred=np.argmax(logit,axis=1)
            print("{} {}: {}".format(name,metric,classification_metrics(pred,lables,metric)))
        # we need more ensemble methods here    
        if method=="simple_add":
            logits=np.array(logits)
            pred=np.argmax(np.sum(logits,axis=0),axis=1)
            print("ensemble model {}: {}".format(metric,classification_metrics(pred,lables,metric)))
                
def main():    
    args,_=get_args()
    #check_gpu(args)
    cfgs=get_ensemble_cfgs(args)
    train_dataset, valid_dataset, test_dataset, Processor = load_dataset(cfgs[0],test=True)
    plms,prompt_models=[],[]
    for order,item in enumerate(cfgs):
       plm_model, plm_tokenizer, plm_config, plm_wrapper_class = load_plm_from_config(item) 
       plm= PLM(plm_model, plm_tokenizer, plm_config, plm_wrapper_class)
       plms.append(plm)
       if item.task == "classification":
            template = load_template(config=item, model=plms[order].plm, tokenizer=plms[order].plm_tokenizer, plm_config=plms[order].plm_config)
            verbalizer = load_verbalizer(config=item, model=plms[order].plm, tokenizer=plms[order].plm_tokenizer, plm_config=plms[order].plm_config, classes=Processor.labels)
            prompt_model = PromptForClassification(plm_model, template, verbalizer, freeze_plm = item.plm.optimize.freeze_para)
            test_dataloader = build_dataloader(test_dataset, template, plms[order].plm_tokenizer, plms[order].plm_wrapper_class, item, "test")
            prompt_models.append(PROMPT_MODLE(prompt_model,test_dataloader))
    # TODO: a latent bug may lay in here. 
    # consume too much resources and prone to trigger cuda error: out of memory
    for order in range(len(prompt_models)):
        try:
            checkpoint_path="../experiments/logs/"+ args.ensemble[order] + "/seed-" + args.seed[order] + "/checkpoints/"
            state_dict = torch.load(checkpoint_path+"best.ckpt", pickle_module = dill, map_location = "cpu")
            prompt_models[order].model.load_state_dict(state_dict['state_dict'])
        except FileNotFoundError:
            raise Exception("checkpoint not found!")
        else:
            print("model {}, seed {} load correctly".format(args.ensemble[0],args.seed[order]))
        prompt_models[order].model.to('cuda:{}'.format(order))            
    logits,lables=multi_forward(prompt_models) 
    ensemble(args,cfgs,"simple_add",logits,lables)    
    
          
if  __name__=='__main__':
    main()

