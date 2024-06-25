# # %%
# %pip install transformers

# # %%
# %pip install awswrangler

# %%
import awswrangler as wr

# %%
import pandas as pd
import psutil
import numpy as np
import evaluate
import torch


# %%
# s3_path_training_data = "s3://cleo-data-science/transaction_enrichment/experimental_data/caste/cross_encoder/xenc1/train_cons_2024-05-15_2024-05-18_1.parquet"
# df_train = wr.s3.read_parquet(s3_path_training_data)
# print(df_train.shape[0])
# #-------------
# s3_path_test_data = "s3://cleo-data-science/transaction_enrichment/experimental_data/caste/cross_encoder/xenc1/test_cons_2024-05-15_2024-05-18_1.parquet"
# df_test = wr.s3.read_parquet(s3_path_test_data)
# print(df_test.shape[0])


from datasets import Dataset, load_dataset, load_from_disk
# ds_train = Dataset.from_pandas(df_train[[label_col,text_col]])
# ds_test = Dataset.from_pandas(df_test[[label_col,text_col]])
ds_train = load_from_disk('/home/sagemaker-user/data/xencdups2/train_dataset')
#ds_test = load_from_disk('/home/sagemaker-user/data/xencdups2/test_dataset')
ds_val = load_from_disk('/home/sagemaker-user/data/xencdups2/val_dataset')


print(f"Length of training dataset {len(ds_train)}")
print(f"Length of validation  dataset {len(ds_val)}")
# %%
label_col = 'label'
text_col = 'text'

# ds_train.save_to_disk('/home/sagemaker-user/data/xencdups/train_dataset')
# ds_test.save_to_disk('/home/sagemaker-user/data/xencdups/test_dataset')

# %% [markdown]
# # Load duplicates_names 

# %%

df_duplicates_candidates = wr.s3.read_csv("s3://cleo-data-science/transaction_enrichment/experimental_data/caste/embedding_predictions/model9/checkpoint-7800/duplicate_merchant_matching_candidates_N_10_cons_2024-05-15_2024-05-18_1.csv", index_col=None)


# %%
df_duplicates_candidates.head()

# %%
try:
    df_duplicates_candidates.drop('Unnamed: 0', axis=1, inplace=True)
except:
    pass

# %%

# %% [markdown]
# ### Load raw data 


# %%

# Process.memory_info is expressed in bytes, so convert to megabytes
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

# %%
ds_train[0]

# %%

# Process.memory_info is expressed in bytes, so convert to megabytes
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

# %%
len(ds_train['text'])

# %%
ds_train[0]

# %%

# %%
print(f" percentage of positive labels in train {np.array(ds_train[label_col]).sum()/len(ds_train)}")

# %%
print(f"percentage of positive labels in val{np.array(ds_val[label_col]).sum()/len(ds_val)}")

# %% [markdown]
# # Load model

# %%
torch.cuda.empty_cache()


# %%
from transformers import AutoTokenizer

model_name = "google-bert/bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)




# %%
#tokenized_ds_train = ds_test.map(tokenize_function, batched=True)
tokenized_ds_train = ds_train.map(tokenize_function, batched=True)
dataset_head = tokenized_ds_train.take(5)
list(dataset_head)

# %%
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

# %%
tokenized_ds_val = ds_val.map(tokenize_function, batched=True)
#tokenized_ds_test = ds_test.map(tokenize_function, batched=True)

# %%
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# %%
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

# %%


metric = evaluate.load("accuracy")
#metric = evaluate.load("precision")

# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# %%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="/home/sagemaker-user/models/xenc3", evaluation_strategy="steps", \
    num_train_epochs=1, per_device_train_batch_size=16, per_device_eval_batch_size=16, \
         eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=100,
    logging_steps=500)

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds_train,
    eval_dataset=tokenized_ds_val
    #compute_metrics=compute_metrics,
)

# %%
trainer.train()

# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# ---- evaluate on validation data 
# -------------------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained("/home/sagemaker-user/models/xenc1/checkpoint-1500", num_labels=1)

# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    #train_dataset=tokenized_ds_train,
    eval_dataset=tokenized_ds_val,
    compute_metrics=compute_metrics,
)

# %%
print(f" samples in validation {len(tokenized_ds_val)}")

predictions = trainer.predict(tokenized_ds_val)
df_predictions = pd.DataFrame(predictions[0], columns=['predictions'])
df_predictions['label'] = ds_val['label']
df_predictions['transaction_id'] = ds_val['transaction_id']
#df_predictions.set_index('transaction_id', inplace=True)
df_predictions

path_s3_output = "s3://cleo-data-science/transaction_enrichment/experimental_data/caste/predictions_dedup/xenc2"
fname_s3_out = path_s3_output+"val_dataset"
wr.s3.to_parquet(df_predictions, fname_s3_out)
print(fname_s3_out)



# %%
th = 0.5
pred_acc = (1*(df_predictions['predictions']>0.5) == df_predictions['label']).sum()/df_predictions.shape[0]
true_positives = df_predictions[(df_predictions['label']==1) & (df_predictions['predictions']>=th)].shape[0]
false_positives  = df_predictions[(df_predictions['label']==0) & (df_predictions['predictions']>=th)].shape[0]
pred_precision = true_positives/(true_positives+false_positives)
false_negatives = df_predictions[(df_predictions['label']==1) & (df_predictions['predictions']<th)].shape[0]
print(pred_precision)
pred_recall = true_positives/(true_positives + false_negatives)
print(f"Precision {pred_precision} Recall {pred_recall}")
