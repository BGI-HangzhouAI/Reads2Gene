import json 
import pandas as pd


def prepare_dataset(file_path,data_type):
    with open(file_path) as f:
        data = json.load(f)
    
    train_data = pd.DataFrame(columns=['data_source', 'prompt', 'ability','reward_model','extra_info'])
    
    data_source = []
    prompt = []
    ability = []
    reward_model = []
    extra_info = []
    
    SYSTEM_PROMPT = """
    Respond in the following format:
    <answer>
    ...
    </answer>
    """
    
    for idx in range(len(data)):
        content = data[idx]
        question = content['question']
        answer = content['answer']
        dna_sequences = content['dna_sequences']
        data_source.append('copypaste')
        prompt.append([
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': '[' + ','.join(dna_sequences) + ']'+"""\n"""+ question +" /no_think"}
            ])
        ability.append('dna_sequence_assembly')
        reward_model.append({"style":"rule",
                             "ground_truth":answer})
        extra_info.append({"split":data_type,
                           "dna_sequence":dna_sequences,
                           "index":idx})
    
    
    train_data['data_source'] = data_source
    train_data['prompt'] = prompt
    train_data['ability'] = ability
    train_data['reward_model'] = reward_model
    train_data['extra_info'] = extra_info
    print(train_data.head())
    train_data.to_parquet('/path/demo/gene'+data_type+'.parquet')



prepare_dataset(file_path='/path/demo/train_2000_copypaste_v8.json',
                data_type='gene_copypaste_train_v8')
prepare_dataset(file_path='/path/demo/val_100_copypaste_v8.json',
                data_type='gene_copypaste_val_v8')
