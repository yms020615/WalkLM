import torch
import random
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from tqdm import tqdm
import sys
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForCausalLM
from torch.utils.data import Dataset

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# Set device for training and inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "google/flan-t5-xl" # "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name) #, token='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')
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
# model = AutoModelForCausalLM.from_pretrained(model_name,
# quantization_config=bnb_config,
# device_map="auto",
# token='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')

'''

def generate_edge_with_t5(current_node, next_node, tokenizer, model, device):
    input_text = f"""Write the connection words such as conjunction, preposition, or linking verb that naturally connect the two disease-related words.
{current_node} ___ {next_node}
Answer: """
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(input_ids, max_length=3, temperature=0.6, num_beams=5, early_stopping=True)
    try:
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split()[0]
    except:
        return random.choice(['and', 'in', 'causing', 'with'])
    # print(input_text, generated_text, '\n')
    return generated_text

op1, op2, op3 = [], [], []

with open('./PubMed/node.dat', 'r') as original_meta_file:
    for line in original_meta_file:
        temp1, temp2, temp3 = line.split('\t')
        op1.append(int(temp1))
        op2.append(temp2)
        op3.append(temp3[:-1])

G = [[] for i in range(len(op3))]

with open('./PubMed/link.dat', 'r') as original_meta_file:
    for line in original_meta_file:
        start, end, edge_type, edge_class = line.split('\t')
        G[int(start)].append([int(end), int(edge_type)])

line_idx = op1
rand = random.Random()
patient_patient_path = []
alpha = 0.15
path_length = 40000
path_num = 10000

dic = {}
for line in tqdm(range(path_num)):
    temp_path = []
    start_path = rand.choice(line_idx)
    temp_path.append([start_path, -1])
    dic[start_path] = 1

    for i in range(path_length):
        cur = temp_path[-1][0]

        if len(G[cur]) > 0:
            if rand.random() >= alpha:
                cur_path = rand.choice(G[cur])
                temp_path.append(cur_path)
                dic[cur_path[0]] = 1
            else:
                break
        else:
            break

    if len(temp_path) >= 2:
        patient_patient_path.append(temp_path)

with open('./PubMed/input_for_gpt4.txt', 'w') as f:
    for i in tqdm(range(len(patient_patient_path))):
        print(op2[patient_patient_path[i][0][0]], '___', op2[patient_patient_path[i][1][0]], end='', file=f)

        for j in range(1, len(patient_patient_path[i])-2):
            print(' ' + '___', op2[patient_patient_path[i][j+1][0]], end='', file=f)

        if len(patient_patient_path[i]) > 2:
            print(' ' + '___', op2[patient_patient_path[i][-1][0]], end='', file=f)

        print('\n', end='', file=f)

with open('./PubMed/output_t5.txt', 'w') as f:
    for i in tqdm(range(len(patient_patient_path))):
        print(op2[patient_patient_path[i][0][0]], generate_edge_with_t5(op2[patient_patient_path[i][0][0]], op2[patient_patient_path[i][1][0]], tokenizer, model, device), op2[patient_patient_path[i][1][0]], end='', file=f)

        for j in range(1, len(patient_patient_path[i])-2):
            print(' ' + generate_edge_with_t5(op2[patient_patient_path[i][j][0]], op2[patient_patient_path[i][j+1][0]], tokenizer, model, device), op2[patient_patient_path[i][j+1][0]], end='', file=f)

        if len(patient_patient_path[i]) > 2:
            print(' ' + generate_edge_with_t5(op2[patient_patient_path[i][-2][0]], op2[patient_patient_path[i][-1][0]], tokenizer, model, device), op2[patient_patient_path[i][-1][0]], end='', file=f)

        print('\n', end='', file=f)

with open('./PubMed/output_t5.txt', 'r') as file:
    corpus = [line.rstrip('\n') for line in file.readlines()]
    print(len(corpus))

train_text, val_text = train_test_split(corpus, test_size=0.15, random_state=42)

# Save the training and validation data
with open('./PubMed/train_corpus_t5.txt', 'w') as file:
    for paragraph in train_text:
        file.write(paragraph + '\n')

with open('./PubMed/val_corpus_t5.txt', 'w') as file:
    for paragraph in val_text:
        file.write(paragraph + '\n')

'''
'''
class CustomTextDataset(Dataset):
    def __init__(self, tokenized_data_pairs):
        self.tokenized_data_pairs = tokenized_data_pairs

    def __len__(self):
        return len(self.tokenized_data_pairs)

    def __getitem__(self, idx):
        return self.tokenized_data_pairs[idx]

def load_custom_dataset(input_file, target_file, tokenizer, max_length=512):
    with open(input_file, 'r', encoding='utf-8') as f_input, open(target_file, 'r', encoding='utf-8') as f_target:
        input_texts = f_input.readlines()
        target_texts = f_target.readlines()

    assert len(input_texts) == len(target_texts), "Input and target files must have the same number of lines."

    tokenized_pairs = []
    for input_text, target_text in zip(input_texts, target_texts):
        tokenized_input = tokenizer(input_text.strip(), max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
        tokenized_target = tokenizer(target_text.strip(), max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
        tokenized_pairs.append({'input_ids': tokenized_input.input_ids.squeeze(), 'attention_mask': tokenized_input.attention_mask.squeeze(), 'labels': tokenized_target.input_ids.squeeze()})

    return CustomTextDataset(tokenized_pairs)

input_file = 'PubMed/input_for_gpt4.txt'
target_file = 'PubMed/output_ablation.txt'

tokenized_data_pairs = load_custom_dataset(input_file, target_file, tokenizer)
train_data, eval_data = train_test_split(tokenized_data_pairs, test_size=0.15, random_state=42)
train_dataset = {"train": train_data}
eval_dataset = {"validation": eval_data}
# decoder_input_ids = tokenizer("Write the connection words such as conjunction, preposition, or linking verb that naturally connect the two disease-related words.", return_tensors="pt").input_ids

config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=['q', 'v'],
    bias="none",
    lora_dropout=0.1,
    task_type="SEQ_2_SEQ_LM" # "CAUSAL_LM"
)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

model = get_peft_model(model, config)
print_trainable_parameters(model)

model = accelerator.prepare_model(model)

# Training arguments
training_args = Seq2SeqTrainingArguments( # TrainingArguments( 
    output_dir="model/t5_ablation",
    overwrite_output_dir=False,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    # bf16=True,
    # optim="paged_adamw_8bit",
    weight_decay=0.1,
    # lr_scheduler_type='linear',
    load_best_model_at_end=True,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    learning_rate=0.0002
)

data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False
    )

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset["train"],
    eval_dataset=eval_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train and save the model
trainer.train()
trainer.save_model("./t5_ablation")

# Generate embeddings
model.eval()
'''
'''
# MLM simulation with T5
def prepare_t5_input_and_target(text, masked_indices, tokenizer, mask_placeholder="<extra_id_0>"):
    tokens = text.split()

    target_words = [tokens[idx] for idx in masked_indices]
    for idx in sorted(masked_indices, reverse=True):
        tokens[idx] = mask_placeholder

    masked_text = " ".join(tokens)
    target_text = " ".join(target_words)  
    return masked_text, target_text

def predict_with_t5(masked_text, true_text, tokenizer, model, device):
    input_ids = tokenizer(masked_text, return_tensors="pt").input_ids.to(device)

    outputs = model.generate(input_ids=input_ids)

    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_text

masked_text, true_text = prepare_t5_input_and_target(
    "5-FUR is observed in Daidzin is detected in Vav1",
    [1, 2, 3, 5, 6, 7],
    tokenizer
)
predicted_text = predict_with_t5(masked_text, true_text, tokenizer, model, device)

print(f"Masked text: {masked_text}")
print(f"True text: {true_text}")
print(f"Predicted text: {predicted_text}")
'''

# Re-load the tokenizer and model for generating embeddings
model_name = "t5_ablation"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

op1, op2, op3 = [], [], []
with open('./PubMed/node.dat', 'r') as original_meta_file:
    for line in original_meta_file:
        temp1, temp2, temp3 = line.split('\t')
        op1.append(temp1)
        op2.append(temp2)
        op3.append(temp3)

def get_word_embeddings(enc_word, dec_word, device):
    input_ids = tokenizer(enc_word, return_tensors="pt").input_ids
    labels = tokenizer(dec_word, return_tensors="pt").input_ids
    input_ids, labels = input_ids.to(device), labels.to(device)
    
    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels)
        # Use the last hidden state
        embeddings = output.encoder_last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

# Initialize embedding matrix
sample_embedding = get_word_embeddings('hello', 'hello', device)
emb = np.zeros((len(op2), sample_embedding.shape[1]))

# Generate embeddings for each entity
for i, word in tqdm(enumerate(op2), total=len(op2)):
    emb[i] = get_word_embeddings(word, word, device)

# Save embeddings to file
with open('./PubMed/emb_t5_ablation.dat', 'w') as file:
    file.write('pubmed\n')
    for i in tqdm(range(len(op2)), total=len(op2)):
        file.write(f'{i}\t')
        file.write(' '.join(map(str, emb[i])))
        file.write('\n')