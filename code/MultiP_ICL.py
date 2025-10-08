from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import re
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, fbeta_score, recall_score, precision_score
from pathlib import Path
import os
import json
from transformers.utils import logging
from .PG_utility.BM25_retriever import BM25_retriever

class llama_model:

    def __init__(self, config, data_config, PT_Generator):
    
        model_name_mapping = {"llama3-8b":"meta-llama/Meta-Llama-3-8B-Instruct"}
        
        access_token = "xxx"
        self.model = AutoModelForCausalLM.from_pretrained(model_name_mapping[config["model"]], device_map="auto", token=access_token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_mapping[config["model"]], token=access_token)
            
        self.config = config
        self.data_config = data_config
        self.PT_Generator = PT_Generator
        self.get_output_file_name()

    
    def get_output_file_name(self):

        if self.config["output_path"] != None:

            output_path_mn = self.config["model"]
            output_path_task = self.config["task"]
            kshot = str(self.config["kshot"])
            
            output_file_name = Path(self.config["output_path"], output_path_mn, output_path_task, f"kshot_{kshot}_prediction.csv") #########################################################################
            
            self.output_file_name = output_file_name

                        
    def predict(self, train_df, test_df): 

        print("************", self.config["task"], "************")
        
        
        #test_df = test_df.iloc[:10, :]

        if self.config["kshot"] != 0:
            kshot_retriever = BM25_retriever(self.data_config, train_df)
            kshot_examples = kshot_retriever.extract_kshots(test_df, self.config["kshot"])
        else:
            kshot_examples = ["-100"]*test_df.shape[0]


        print("DATA SIZE:", test_df.shape[0])
        
        if self.output_file_name.exists():
            print("Existed")
            return
            
        predictions = []
        prompt_example = []
        no_logits = []
        yes_logits = []


        for i in range(test_df.shape[0]):
            
            input_text = test_df.loc[i, self.data_config["input_field"]]
            examples = kshot_examples[i]
            prompt = self.PT_Generator.generate_prompt(input_text, examples)

            #######
            messages = [{"role": "user", "content":prompt}]
            logging.set_verbosity_error()
            model_inputs = self.tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt", max_length = 14000).to("cuda")  
            terminators = [self.tokenizer.eos_token_id,self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            #######
            
            if self.config["temperature"] != None:
                output_ = self.model.generate(model_inputs, max_new_tokens=150, eos_token_id=terminators, do_sample=True, top_k = None, temperature=self.config["temperature"], top_p = None)
            else:
                output_ = self.model.generate(model_inputs, max_new_tokens=150, output_scores=True, return_dict_in_generate=True, eos_token_id=terminators, do_sample=False, top_k = None, temperature=None, top_p = None)
                
            logits = output_.scores[0]
            
            ## Note: why 2201 and 9891 below: tokenizer.encode("no")=2201; tokenizer.encode("yes")=9891. If the model sometimes output "Yes", "No", or punctruation at the first genearted token, then it's not precise.
            
            logits_no_yes = torch.softmax(logits[0, [2201,9891]], dim=-1).detach().cpu().numpy() 
            
            no_logits.append(logits_no_yes[0])
            yes_logits.append(logits_no_yes[1])  
            response = output_.sequences[0][model_inputs.shape[-1]:]
            
            prediction = self.tokenizer.decode(response, skip_special_tokens=True)
            predictions.append(prediction)
            
            if (i == 0) or (i == 1):
                print(prompt)
                print("########prediction:", prediction)
                prompt_example.append(prompt)
            else:
                prompt_example.append("0") 
            
            if i%100 == 1:
                print(i)
        
        assert len(predictions) == test_df.shape[0]
        
        print("Finished!") 
        
        test_df["predictions"] = predictions
        test_df["prompt_example"] = prompt_example
        test_df["no_logits"] = no_logits
        test_df["yes_logits"] = yes_logits
        
        if self.config["output_path"] != None:
            test_df.to_csv(self.output_file_name, index = False)
