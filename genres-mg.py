import torch
import nlp
from tqdm import tqdm
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

model_run_name = 'baseline_mg_w_pretrain_20_cl_fib'
BEAM_SIZE = 1

spm_path = '/home/amharris/aug/vocab/dl4se_vocab.model'
config_file = f'/home/amharris/aug/outmodels/{model_run_name}/config.json'
config = T5Config.from_json_file(config_file)
tokenizer = T5Tokenizer.from_pretrained(spm_path)

def add_eos_to_examples(example):
    example['input_text'] = 'generate mutant: %s </s>' % example['buggy']#.lower()
    example['target_text'] = '%s </s>' % example['fixed']#.lower()
    return example


def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, max_length=512)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=512)

    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings

valid_dataset = nlp.load_dataset('mg_dataset_script.py', split=nlp.Split.TEST)


# map add_eos_to_examples function to the dataset example wise 
valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)

# map convert_to_features batch wise
valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)


columns = ['input_ids', 'target_ids', 'attention_mask','target_attention_mask']
valid_dataset.set_format(type='torch', columns=columns)

BATCH_SIZE = 8
dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE)

import pandas as pd

df = pd.read_csv("/home/amharris/aug/mg_test.tsv",header=None,sep='\t')

references=[]

for item in df[1]:
  references.append(item)

inputs=[]

for item in df[0]:
  inputs.append(item)

inputs[0], references[0]

CUDA = torch.device("cuda")

finetuned_model_path = f'/home/amharris/aug/outmodels/{model_run_name}/pytorch_model.bin'

model = T5ForConditionalGeneration.from_pretrained(
        finetuned_model_path,
        config=config
        ).to(CUDA)
        
model.eval()

from tqdm import tqdm

predictions = []



torch.cuda.empty_cache()

for batch in tqdm(dataloader):

      outs = model.generate(
                          input_ids=batch['input_ids'].to(CUDA),
                          attention_mask=batch['attention_mask'].to(CUDA),
                          num_beams=BEAM_SIZE, 
                          max_length=128,
                          num_return_sequences=BEAM_SIZE, 
                          early_stopping=True
                          )
    

    
      outs = [tokenizer.decode(ids, skip_special_tokens=True)  for ids in outs]
      predictions.extend(outs)

pred_refined = []
for pred in predictions:
    if len(pred)>=2:
      if pred[0]=='"':
          pred = pred[1:]
      if pred[-1]=='"':
          pred = pred[:-1]
    pred_refined.append(pred)
    
len(pred_refined),len(predictions)

counter_pred = 0

mispred_list = []

sanity_check_list = []

idx = 0

len_prediction=(len(pred_refined))


for i in range(0, len_prediction, BEAM_SIZE):

    items_to_analyze = pred_refined[i:i+BEAM_SIZE]
    target_item = ''.join(references[idx].split(' '))
    
    flag_perfect = False

    
    fpred=open(f'/home/amharris/aug/preds/{model_run_name}/prediction_@{BEAM_SIZE}.txt','a+')
    fpred.write('************************************\n')
    fpred.write('[+] input: {}\n'.format(inputs[idx]))
        

    for pred in items_to_analyze:
        
        pred_ref = ''.join(pred.split(' '))
        
        fpred.write('[*] target: {}\n'.format(references[idx]))
        fpred.write('[-] pred:  {}\n\n'.format(pred))

        if pred_ref == target_item and not flag_perfect:
            counter_pred+=1
            sanity_check_list.append(pred)

            with open(f'/home/amharris/aug/preds/{model_run_name}/perfect_@{BEAM_SIZE}.txt','a+') as fwrite:
                fwrite.write('[+] input: {}\n'.format(inputs[idx]))
                fwrite.write('[*] target: {}\n'.format(references[idx]))
                fwrite.write('[-] pred:  {}\n\n'.format(pred))
            
            flag_perfect = True
        
        else:
          mispred_list.append(pred)
      
    fpred.write('************************************\n')
        
    idx += 1

fpred.close()
print(f"Run: {model_run_name} / Beam: {BEAM_SIZE}")
print('% of perfect predictions: ',(counter_pred/len(references))*100 )
print(counter_pred)