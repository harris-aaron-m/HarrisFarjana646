import pandas as pd

# The predictions file:
preds_file = 'preds/pt_cl_bfm_sched14d/prediction_@1.txt'
# What to save the results as:
save_name = 'pt_cl_bfm_sched14d_pred@1.tsv'

preds = []
inputs = []
targets = []
match = []

with open(preds_file, 'r') as f:
    for line in f:
        o_line = line
        line = ''.join(line.split(' '))
        if line[0] == '*':
            currentInput = ''
            currentTarget = ''
            currentPred = ''
        if line[0:3] == '[+]':
            currentInput = ''.join(line[3:].split(':')[1:]).replace('\n','')
            original_input = ' '.join(o_line[3:].split(':')[1:]).replace('\n','')
        if line[0:3] == '[*]':
            currentTarget = ''.join(line[3:].split(':')[1:]).replace('\n','')
            original_target = ' '.join(o_line[3:].split(':')[1:]).replace('\n','')
        if line[0:3] == '[-]':
            currentPred = ''.join(line[3:].split(':')[1:]).replace('\n','')
            original_pred = ' '.join(o_line[3:].split(':')[1:]).replace('\n','')
            preds.append(original_pred)
            inputs.append(original_input)
            targets.append(original_target)
            if currentPred == currentTarget:
                match.append(1)
            else:
                match.append(0)

df = pd.DataFrame({'input':inputs,'target':targets,'pred':preds,'match':match})
df.to_csv(save_name, sep='\t')
