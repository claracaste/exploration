# %% [markdown]
# - look into architechture modification and add numerical values as concatenation

# %% [markdown]
# https://sbert.net/docs/sentence_transformer/training_overview.html#dataset-format

# %%
# %pip install accelerate==0.27.2
# %pip install sentence-transformers
# %pip install umap-learn
#%pip install cleodata --extra-index-url "https://aws:$(aws codeartifact get-authorization-token --domain meetcleo --domain-owner 878877078763 --query authorizationToken --output text)@meetcleo-878877078763.d.codeartifact.us-east-1.amazonaws.com/pypi/meetcleo-releases/simple/"

# %%
# %pip install awswrangler
# %pip install --upgrade sentence_transformers

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
from sentence_transformers import SentenceTransformer, models
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
from sentence_transformers import losses
from sklearn.model_selection import train_test_split

# %%
s3_path = "s3://cleo-data-science/transaction_enrichment/experimental_data/caste/trx-merchant-pair/trans_2024-05-14_2024-05-14_top_2001.parquet"

df_data_raw = wr.s3.read_parquet(path=s3_path)
df_data_raw.shape

# %%
print(torch.cuda.device_count())

# %%
df_data_raw.head()

# %%
#%pip install s3fs --upgrade
#import s3fs

# %%
df_ , df_test, y_, y_test = train_test_split(df_data_raw, df_data_raw['true_label'], test_size = 0.05, random_state=1)
df_train, df_val, y_train, y_val = train_test_split(df_, df_['true_label'], test_size = 0.1, random_state=1)
df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
print(df_train.shape[0], df_val.shape[0], df_test.shape[0])

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
directory = "models/model3"

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
    output_dir="models/model3",
    # Optional training parameters:
    num_train_epochs=3,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=200,
    logging_steps=100,
    run_name="test1",  # Will be used in W&B if `wandb` is installed
)

# %%
print(ds_val[0])

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
trainer.train()

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
model.save_pretrained("models/model3-final")

# %%
print(f"Finished model training")


