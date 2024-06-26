# %%
import pandas as pd
import numpy as np


# %%
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from io import StringIO
import re
# from fastparquet import ParquetFile
# import s3fs
# from cleodata.utils.secrets import get_secret
# from cleodata.sources.sync.sync import SyncDataSource
# boto3.setup_default_session(profile_name='DataScientist-878877078763')
# redshift_source = SyncDataSource("data_exploration", use_redshift=True, redshift_cluster="cleo-production-redshift", redshift_db="cleo")

# %%
from sagemaker import get_execution_role
role = get_execution_role()

# %%
def read_from_s3(path):
    """Read parquet files and combine them into a single dataframe"""
    fs = s3fs.core.S3FileSystem()
    all_paths_from_s3 = fs.glob(path=f"{path}*.parquet")

    if len(all_paths_from_s3) > 0:
        s3 = s3fs.S3FileSystem()
        fp_obj = ParquetFile(
            all_paths_from_s3, open_with=s3.open
        )  # use s3fs as the filesystem
        data = fp_obj.to_pandas()
        return data
    elif len(all_paths_from_s3)==1:
        return pd.read_parquet(all_paths_from_s3[0])
    else:
        print(f"Nothing found")
        print(f"paths from a{all_paths_from_s3}")
    
def read_csv_s3(bucket, key):
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(obj['Body'])
        return df
    except ClientError as ex:
        if ex.response['Error']['Code'] == 'NoSuchKey':
            print("Key doesn't match. Please check the key value entered.")


# %% [markdown]
# # Create positive and negative data pairs

# %%
# %pip install pyarrow
# %pip install fastparquet
# %pip install awswrangler

# %%
import awswrangler as wr

# %%
path_new_file = "s3://cleo-data-science/transaction_enrichment/experimental_data/caste/processed_duplicates/trans_2024-05-15_2024-05-15"
df_data_raw = wr.s3.read_parquet(path=path_new_file)

# %%
df_data_raw.head()

# %%
# path_new_file = "s3://cleo-data-science/transaction_enrichment/experimental_data/caste/processed/trans_2024-05-14_2024-05-14"
# df_data_raw = wr.s3.read_parquet(path=path_new_file)

# %%
df_data_raw.head()

# %%
#df_data_raw[df_data_raw['merchant_name_combined'] == 'Myfico']

# %%
path_file_out = path_new_file.split('processed_duplicates/')[0]+'trx-merchant-pair/'+path_new_file.split('processed_duplicates/')[1]
path_file_out

# %%
df_data_raw.head()

# %%
print(df_data_raw['sentence'][1000050], df_data_raw['merchant_name_combined'][1000050])

# %%
df_data_raw['num_words'] = df_data_raw['description_combined_processed'].apply(lambda x: len(x.split(' ')))

# %%
df_data_raw['num_words'].describe()

# %%
df_data_raw['num_words'].max()*3

# %%
df_data_raw[df_data_raw['num_words']==41]

# %%

df_data_raw.loc[1600443,'description_combined_processed']

# %%
df_data_raw.loc[1600443,'description_combined']

# %%
df_data_raw.loc[1600443,'merchant_name_combined']

# %%
df_data_raw['len_sentence'] = df_data_raw['description_combined_processed'].apply(lambda x: len(x.split(' ')))

# %%
df_data_raw['len_sentence'].max()

# %%
df_data_raw.head()

# %%
df_data_raw['merchant_name_combined_len'] = df_data_raw['merchant_name_combined'].apply(lambda x: len(x))
df_data_raw = df_data_raw[df_data_raw['merchant_name_combined_len']>=1]
df_data_raw.drop('merchant_name_combined_len', axis=1, inplace=True)


# %%
print(df_data_raw[df_data_raw['len_sentence']==df_data_raw['len_sentence'].max()]['description_combined'])

# %%
df_merchant_volume = df_data_raw['merchant_name_combined'].value_counts(dropna=False).to_frame()
df_merchant_volume

# %%
n_tot_trans  = df_data_raw.shape[0]
df_merchant_volume['perc_traffic'] = df_merchant_volume['count']/n_tot_trans
df_merchant_volume.sort_values(by='perc_traffic', ascending=False)
df_merchant_volume['cumulative_traffic'] = df_merchant_volume['perc_traffic'].cumsum()
df_merchant_volume

# %%
df_merchant_volume

# %%
import matplotlib.pyplot as plt
plt.plot(np.arange(df_merchant_volume.shape[0]),df_merchant_volume['cumulative_traffic'],'.')
plt.xlim([0,4000])

# %%
df_merchant_volume.reset_index(drop=False,inplace=True)
df_merchant_volume[0:10]

# %%
top_n = 10000
merchants_top_n = df_merchant_volume['merchant_name_combined'][0:top_n].tolist()
print(len(merchants_top_n))
merchants_top_n

# %%
15*1000/3600

# %%
20*50/3600

# %%
166/60

# %%
18*60*10*5/3600

# %%
# !pwd
# !ls -ltr /opt/amazon/sagemaker/sagemaker-code-editor-server-data/data/User/History
# !more /opt/amazon/sagemaker/sagemaker-code-editor-server-data/data/User/History/14a6abb3/entries.json
# !touch /opt/amazon/sagemaker/sagemaker-code-editor-server-data/data/User/History/
# !touch /home/sagemaker-user/toto
# !echo "abdc" > /home/sagemaker-user/toto
# !more /home/sagemaker-user/toto

# %%
#!pip install path

# %%
from path import Path




# %%
n_pairs = 10000
max_rows_per_type = 3*n_pairs
  
# %%
top_tx_merchants = df_merchant_volume[df_merchant_volume['count']>max_rows_per_type]['merchant_name_combined'].tolist()
top_tx_merchants


# %%
df_top = df_data_raw[df_data_raw['merchant_name_combined'].isin(top_tx_merchants)]
df_not_top = df_data_raw[~df_data_raw['merchant_name_combined'].isin(top_tx_merchants)]
sampled_df = df_top.groupby('merchant_name_combined').apply(lambda x: x.sample(min(len(x), max_rows_per_type))).reset_index(drop=True)

# %%
df_all_sampled = pd.concat([sampled_df,df_not_top], axis=0)

# %%
#This step needs to be parallelized
df_all = pd.DataFrame()
for i,merchant in enumerate(merchants_top_n):
  #print(i,merchant)
  df_s1 = df_all_sampled[df_all_sampled['merchant_name_combined']==merchant][['transaction_id','sentence','sentence2',\
    'description_combined_processed',\
    'merchant_name_combined','is_duplicate','original_merchant_name_combined','amount']]
  df_s1.rename(columns={'merchant_name_combined':'true_merchant_name_combined'}, inplace=True)

  df_s2 = df_all_sampled[df_all_sampled['merchant_name_combined']!=merchant][['transaction_id','merchant_name_combined']]
  
  df_negative = pd.concat([df_s1.sample(min(df_s1.shape[0],2*n_pairs),random_state=1)[['transaction_id','sentence','sentence2',\
    'description_combined_processed','true_merchant_name_combined','is_duplicate','original_merchant_name_combined','amount']].reset_index(drop=True), \
                           df_s2.sample(min(df_s1.shape[0],2*n_pairs), random_state=1)[['merchant_name_combined']].reset_index(drop=True)], axis=1 , ignore_index=True)
  df_negative['true_label'] = 0
  df_negative['label'] = 0 + np.abs(np.random.normal(0,0.2,[1,df_negative.shape[0]])[0])
  df_negative.columns = ['transaction_id','sentence','sentence2','description_combined_processed','true_merchant_name_combined',\
   'is_duplicate','original_merchant_name_combined','amount', 'merchant_name_combined','true_label','label']
  df_positive = df_s1.sample(min(df_s1.shape[0],n_pairs), random_state=1)
  df_positive['true_label'] = 1
  df_positive['label'] = 1 - np.abs(np.random.normal(0,0.2,[1,df_positive.shape[0]])[0])
  df_positive['merchant_name_combined'] = df_positive['true_merchant_name_combined']
  df_all = pd.concat([df_all, df_negative, df_positive[['transaction_id','sentence','sentence2','description_combined_processed',\
    'true_merchant_name_combined','is_duplicate','original_merchant_name_combined','amount','merchant_name_combined','true_label','label']] ], axis=0)
  #!echo "abdc" > /home/sagemaker-user/toto
  if i%300 == 0:
    print(i,merchant)
    Path("/opt/amazon/sagemaker/sagemaker-code-editor-server-data/data/User/History/reset_timer.txt").touch()
  if i%1000 == 0:
    print('saving')
    print(i)
    s3_path_out = path_file_out+'_top_'+str(top_n)+'_'+str(i)+'.parquet'
    #s3_path_out = "s3://cleo-data-science/transaction_enrichment/experimental_data/caste/trx-merchant-pair/sample.parquet"
    #s3_path_out = "s3://cleo-data-science/transaction_enrichment/experimental_data/caste/trx-merchant-pair/transactions_2024-05-20_2024-05-20_"+str(i)+".parquet"
    #s3_path_out = "s3://cleo-data-science/transaction_enrichment/experimental_data/caste/trx-merchant-pair/sm_test.parquet"
    #df_all.to_parquet(s3_path_out, engine='pyarrow')
    wr.s3.to_parquet(df = df_all, path =s3_path_out, dataset=True )

    print(i, s3_path_out)


s3_path_out = path_file_out+'_top_'+str(top_n)+'.parquet'
wr.s3.to_parquet(df = df_all, path =s3_path_out, dataset=True )

#df_all.to_parquet(s3_path_out, engine='pyarrow')


# %%
# s3_path_out = path_file_out+'_top_'+str(top_n)+'.parquet'
# df_all.to_parquet(s3_path_out, engine='pyarrow')
print(f"Finished creating {s3_path_out}")
df_data_pairs = wr.s3.read_parquet(path=s3_path_out)


# %%
Path("/opt/amazon/sagemaker/sagemaker-code-editor-server-data/data/User/History/reset_timer.txt").touch()

# %%
  # s3_path_out = "s3://cleo-data-science/transaction_enrichment/experimental_data/caste/trx-merchant-pair/transactions_2024-05-20_2024-05-20_top5K.parquet"
  # #s3_path_out = "s3://cleo-data-science/transaction_enrichment/experimental_data/caste/trx-merchant-pair/sm_test.parquet"
  # df_all.to_parquet(s3_path_out, engine='pyarrow')

# %%
print("finished")

# %%
df_all.reset_index(drop=True, inplace=True)

# %%
df_all.head()

# %%
df_all.loc[0,'sentence']

# %%
df_all.loc[0,'sentence2']

# %%
# df_all.to_csv("/Users/claracastellanos/Documents/DATA/MERCHANTS/2024_05_20_sample_top5K_pairs.csv")

# %%
print(f"Finished creating file")

# %%
#write this data to s3

# %%
df_all.head(20)


