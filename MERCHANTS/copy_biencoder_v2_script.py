# %% [markdown]
# - look into architechture modification and add numerical values as concatenation

# %% [markdown]
# https://sbert.net/docs/sentence_transformer/training_overview.html#dataset-format

# %%
# %pip install -e

# # %%
# %pip install git+https://github.com/huggingface/transformers


# %%
#!cp /opt/conda/lib/python3.10/site-packages/fsspec-2024.6.0.dist-info/METADATA /opt/conda/lib/python3.10/site-packages/fsspec-2023.6.0.dist-info/

# # %%
# %pip install --upgrade pip

# # %%
# %pip install accelerate==0.27.2
# %pip install sentence-transformers
# %pip install umap-learn
# %pip install awswrangler
# %pip install --upgrade sentence_transformers
#%pip install cleodata --extra-index-url "https://aws:$(aws codeartifact get-authorization-token --domain meetcleo --domain-owner 878877078763 --query authorizationToken --output text)@meetcleo-878877078763.d.codeartifact.us-east-1.amazonaws.com/pypi/meetcleo-releases/simple/"

# %%
import pandas as pd
from datetime import datetime
import awswrangler as wr
import boto3
from botocore.exceptions import ClientError
from io import StringIO
#from fastparquet import ParquetFile
#boto3.setup_default_session(profile_name='DataScientist-878877078763')
# from cleodata.utils.secrets import get_secret
#from cleodata.sources.sync.sync import SyncDataSource
#redshift_source = SyncDataSource("data_exploration", use_redshift=True, redshift_cluster="cleo-production-redshift", redshift_db="cleo")

# %%
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
import umap
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import losses
from sklearn.model_selection import train_test_split

# %%
import torch

# %%
print(torch.__version__)

# %%
s3_path = "s3://cleo-data-science/transaction_enrichment/experimental_data/caste/trx-merchant-pair/trans_2024-05-14_2024-05-14_top_2001.parquet"
s3_path = "s3://cleo-data-science/transaction_enrichment/experimental_data/caste/trx-merchant-pair/trans_2024-05-18_2024-05-18_top_1_39000.parquet/"
# consolidate dtata
s3_path = "s3://cleo-data-science/transaction_enrichment/experimental_data/caste/trx-merchant-pair/cons_2024-05-15_2024-05-18_1.parquet/"
s3_path = "s3://cleo-data-science/transaction_enrichment/experimental_data/caste/trx-merchant-pair/cons_2024-05-15_2024-05-18_1.parquet/"

df_data_raw = wr.s3.read_parquet(path=s3_path)
df_data_raw.shape


import re
pattern = r"Cash\sApp\s(?!Transfer)[\w\s]+"
result = re.sub(pattern, "Cash App", 'Cash App Transfer')
result
df_data_raw['original_merchant_name_combined'] = df_data_raw['original_merchant_name_combined'].apply(lambda x: re.sub(pattern, "Cash App", x) )
df_data_raw['merchant_name_combined'] = df_data_raw['merchant_name_combined'].apply(lambda x: re.sub(pattern, "Cash App", x) )
df_data_raw['true_merchant_name_combined'] = df_data_raw['true_merchant_name_combined'].apply(lambda x: re.sub(pattern, "Cash App", x) )


# %%
df_data_raw.head()

# %%
print(df_data_raw['is_duplicate'].sum(), df_data_raw['is_duplicate'].sum()/df_data_raw['true_label'].sum())

# %%
print(torch.cuda.device_count())

# %%
df_data_raw.head()

# %%
#%pip install s3fs --upgrade
#import s3fs

# %%
df_ , df_test, y_, y_test = train_test_split(df_data_raw, df_data_raw['true_label'], test_size = 0.05, random_state=1)
df_train, df_val, y_train, y_val = train_test_split(df_, df_['true_label'], test_size = 0.05, random_state=1)
df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
print(df_train.shape[0], df_test.shape[0], df_val.shape[0])

# %%
text_col = "description_combined_processed"

# %%
df_train['len_sentence'] = df_train[text_col].apply(lambda x: len(x.split(' ')))
print(df_train['len_sentence'].max())

# %%
from sentence_transformers import SentenceTransformer, models
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader

# %%

word_embedding_model = models.Transformer("bert-base-uncased", max_seq_length=128)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# %%
#pd.set_option('display.float_format', lambda x: '%.f' % x)
#estimate maximum tokenized length
df_train['len_sentence'].describe()

# %%
print(model.max_seq_length)
model.max_seq_length = 128
print(model.max_seq_length)

# %%
df_train.head()

# %%
one_sent = df_train[text_col][0]
print(df_train[text_col][1])

# %%
df_train['label'] = df_train['label'].astype('float32')
df_train['true_label'] = df_train['true_label'].astype('float32')
#train_examples = [InputExample(texts = [df_train.loc[i,'sentence'], df_train.loc[i,'merchant_name_combined']], label=df_train.loc[i,'true_label']  ) for i in range(df_train.shape[0])]

# %%
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)

# %%
from torch.utils.data import Dataset, DataLoader
from sentence_transformers.losses import CoSENTLoss


# %%
from datasets import Dataset

ds_train = Dataset.from_pandas(df_train[['true_label',text_col,'merchant_name_combined']])
ds_val = Dataset.from_pandas(df_val[['true_label',text_col,'merchant_name_combined']])
ds_test = Dataset.from_pandas(df_test[['true_label',text_col,'merchant_name_combined']])

ds_train = ds_train.rename_columns({"true_label": "score",text_col:"sentence1", "merchant_name_combined":"sentence2"})
ds_val = ds_val.rename_columns({"true_label": "score",text_col:"sentence1", "merchant_name_combined":"sentence2"})
ds_test = ds_test.rename_columns({"true_label": "score",text_col:"sentence1", "merchant_name_combined":"sentence2"})



# %%
ds_test

# %%
import os
directory = "/home/sagemaker-user/logs"

# Check if the directory already exists
if not os.path.exists(directory):
    # Create the directory
    os.makedirs(directory)
    print("Directory created successfully!")
else:
    print("Directory already exists!")

# %%
import os
directory = "/home/sagemaker-user/models/model9"

# Check if the directory already exists
if not os.path.exists(directory):
    # Create the directory
    os.makedirs(directory)
    print("Directory created successfully!")
else:
    print("Directory already exists!")

# %%
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers import SentenceTransformerTrainer

# %%
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="/home/sagemaker-user/models/model9",
    # Optional training parameters:
    num_train_epochs=3,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    #learning_rate = 5e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    evaluation_strategy="steps", #eval_strategy
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=50,
    logging_steps=100,
    run_name="test1",  # Will be used in W&B if `wandb` is installed
    #load_best_model_at_end= True,
    logging_dir="/home/sagemaker-user/logs",
)

# %%
ds_val[0]

# %%
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=df_val[text_col],
    sentences2=df_val["merchant_name_combined"],
    scores=df_val["true_label"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)

# %%
#dev_evaluator(model)


# %%
train_loss = CoSENTLoss(model)

# %%
# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    loss=train_loss,
    #evaluator=dev_evaluator,
)
trainer.train(resume_from_checkpoint="/home/sagemaker-user/models/model9/checkpoint-5000")

# %%


# %%
# # (Optional) Evaluate the trained model on the test set
# test_evaluator = TripletEvaluator(
#     anchors=df_test["sentence"],
#     positives=df_test["true_merchant_name_combined"],
#     negatives=test_dataset["true_label"],
#     name="pairs-test1",
# )
# test_evaluator(model)

# 8. Save the trained model
model.save_pretrained("/home/sagemaker-user/models/model9final")

# %%
#!ls ./models/model1/checkpoint-600

# %%
#copy 
import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def upload_directory_to_s3(local_directory, bucket_name, s3_directory):
    s3_client = boto3.client('s3')

    for root, dirs, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(s3_directory, relative_path)

            try:
                s3_client.upload_file(local_path, bucket_name, s3_path)
                print(f'Successfully uploaded {local_path} to s3://{bucket_name}/{s3_path}')
            except FileNotFoundError:
                print(f'The file {local_path} was not found')
            except NoCredentialsError:
                print('Credentials not available')
            except PartialCredentialsError:
                print('Incomplete credentials provided')

# Example usage
local_directory = "/home/sagemaker-user/models/model4"
bucket_name = 'cleo-data-science'
s3_directory = "transaction_enrichment/experimental_data/caste/pairs_model/model4"

upload_directory_to_s3(local_directory, bucket_name, s3_directory)


# %%
## 
#model = SentenceTransformer("./models/model1/checkpoint-600")


# %%
list_unique_merchants = df_train['merchant_name_combined'].tolist()
embeddings_merchants = model.encode(list_unique_merchants)
embeddings_merchants.shape

# %%
df_test['pred_merchant'] = None
df_test['pred_prob'] = 0.0
batch_size = 1000

# %%
df_test.shape

# %%
for istart in np.arange(0, df_test.shape[0]+1, batch_size):
    iend = min(df_test.shape[0],istart + batch_size)
    if iend> istart:
        print(istart, iend)
        tx_embeddings = model.encode(df_test['sentence'][istart:iend].tolist())
        similarities = model.similarity(tx_embeddings, embeddings_merchants)
        print(similarities.shape)
        max_vals = torch.max(similarities, axis=1)
        max_probs = max_vals[0]
        ix_max_merchants = max_vals[1]
        predicted_merchant = [list_unique_merchants[i] for i in ix_max_merchants]
        df_test.loc[istart:iend-1,'pred_merchant'] =  predicted_merchant
        df_test.loc[istart:iend-1,'pred_prob'] =  np.array(max_probs)

# %%
ntrue = df_test[df_test['pred_merchant'] == df_test['true_merchant_name_combined']].shape[0]

precision = ntrue/df_test.shape[0]
print(precision)

# %%
ntrue = df_test[(df_test['pred_merchant'] == df_test['true_merchant_name_combined']) & (df_test['pred_prob']>0.8)].shape[0]

accuracy = ntrue/df_test.shape[0]
print(accuracy)

# %%
#tp/tp + fp



# %%
df_no_match = df_test[df_test['pred_merchant'] != df_test['true_merchant_name_combined']]
df_no_match.reset_index(drop=True, inplace=True)
df_no_match

# %%
df_no_match.drop_duplicates(subset = ['true_merchant_name_combined','pred_merchant'])

# %%
df_no_match.loc[13,'sentence']

# %%
df_no_match.loc[269,'sentence']

# %%
# do a precision-recall curve

# %%


# %%


# %%
df_train['true_merchant_name_combined'].value_counts()

# %%
df_train['true_merchant_name_combined'].nunique()

# %%
df_test['true_merchant_name_combined'].value_counts()

# %%
df_val['true_merchant_name_combined'].value_counts()

# %%
df_test[['true_merchant_name_combined','pred_merchant','pred_prob']]

# %%
df_test['pred_prob'].hist();

# %%
df_test[df_test['pred_prob']<0.85][:500]

# %%
df_test.shape

# %%
df_test['sentence'][100:150]

# %%
model.encode(df_test['sentence'][istart:istart+batch_size].tolist())

# %%
df_test.loc[istart:istart+batch_size-1,'pred_merchant'].shape

# %%
df_test.loc[0:100,'label'].shape

# %%
df_test[0:100].shape

# %%
istart+batch_size

# %%

len(predicted_merchant)

# %%
tx_embedding = model.encode(df_test['sentence'][0:2])

# %%
tx_embedding.shape

# %%


# %%


# %%


# %%


# %%


# %%


# %%
df_test['merchant_name_combined']

# %%
matched_merchants = similarities.max(axis=1)
matched_merchants

# %%
similarity_score = matched_merchants[0]
similarity_score

# %%
matched_merchants

# %% [markdown]
# Load model

# %%
print(sentence_transformers.__version__)

# %%
from sentence_transformers import SentenceTransformerModelCardData
from datasets import load_dataset


# %%
# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    "microsoft/mpnet-base",
    model_card_data=SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="MPNet base trained on AllNLI triplets",
    )
)

# 3. Load a dataset to finetune on
dataset = load_dataset("sentence-transformers/all-nli",'pair-score')

# %%
dataset

# %%
train_dataset = dataset['train']

# %%
train_dataset[0]

# %%
#train_dataloader = DataLoader(train_examples, shuffle=True, batch_size = 128)

# %%
len(train_dataloader)

# %%
#train_loss = losses.CosineSimilarityLoss(model)

# %%
train_loss

# %% [markdown]
# ## Tune the model

# %%
model.fit(train_objectives = [(train_dataloader, train_loss)], epochs = 1 , warmup_steps = 100)

# %%
# save model

# %% [markdown]
# ## Evaluation

# %%
trx_descriptions = df_val['sentence'].tolist()
merchants = df_val['merchant_name_combined'].tolist()
trx_embeddings = model.encode(trx_descriptions)
merchants_embeddings = model.encode(merchants)

# %%
print(trx_embeddings.shape)
print(merchants_embeddings.shape)

# %%
import numpy as np

# %%
## look into numba
cosine_similarity(trx_embeddings, merchants_embeddings).shape
df_val['cos_similarity']=np.diag(cosine_similarity(trx_embeddings, merchants_embeddings))

# %%
df_val

# %%
while True:
    pass

# %%
validate_data[['label', 'true_label','cos_similarity']]

# %%


# %%
# reducer = umap.UMAP(n_epochs=400,  n_neighbors=150, min_dist=0.1)
# reducer.fit(merchants_embeddings
# )
# embedding_2d = reducer.transform(merchants_embeddings)
# fig = px.scatter(embedding_2d, x=0, y=1,opacity=0.05, height=500, hover_name=names)
# fig.show()


