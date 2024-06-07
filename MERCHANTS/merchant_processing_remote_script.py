# # %%
# %pip install pyarrow
# %pip install fastparquet
# %pip install awswrangler
# %pip install s3fs


# %%
import pandas as pd
import re
import numpy as np
import awswrangler as wr
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from io import StringIO
import s3fs
from fastparquet import ParquetFile
# from cleodata.utils.secrets import get_secret
# from cleodata.sources.sync.sync import SyncDataSource
# boto3.setup_default_session(profile_name='DataScientist-878877078763')
# redshift_source = SyncDataSource("data_exploration", use_redshift=True, redshift_cluster="cleo-production-redshift", redshift_db="cleo")

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

def list_s3_flies(base_path):
    fs = s3fs.core.S3FileSystem()
    all_paths_from_s3 = fs.glob(path=f"{base_path}*.parquet")
    return all_paths_from_s3


# %%
start_date_s = '2024-05-14'
end_date_s = '2024-05-22'
date_range = pd.date_range(start=start_date_s, end=end_date_s)
# Convert the date range to a list of strings
date_list = date_range.strftime('%Y-%m-%d').tolist()
date_list

# %%
for one_date in date_list:
    start_date = one_date
    end_date = one_date
    print(start_date, end_date)
    #list the files that start with that pattern
    path_file = f"s3://cleo-data-science/transaction_enrichment/experimental_data/caste/raw/trans_{start_date}_{end_date}"
    print(f"Loading {path_file}")
    df_trans = read_from_s3(path_file)
    print(f" Loaded shape {df_trans.shape}")

    path_file_processed = path_file.split('raw/')[0]+'processed/'+path_file.split('raw/')[1]
    print(f" Output {path_file_processed}")
    # Coalescing merchant names

    #replace None with null
    #replace Nan
    # Replace empty spaces, None, and strings with only spaces with NaN
    df_trans['merchant_name'] = df_trans['merchant_name'].replace(r'^\s*$', np.nan, regex=True)
    df_trans['merchant_name_plaid'] = df_trans['merchant_name_plaid'].replace(r'^\s*$', np.nan, regex=True)


    df_trans['merchant_name'] = df_trans['merchant_name'].replace('None',None)
    df_trans['merchant_name'] = df_trans['merchant_name'].replace('',None)
    df_trans['merchant_name'] = df_trans['merchant_name'].replace(' ',None)
    df_trans['merchant_name_plaid'] = df_trans['merchant_name_plaid'].replace('', None)
    df_trans['merchant_name_plaid'] = df_trans['merchant_name_plaid'].replace(' ', None)
    df_trans['merchant_name'] = df_trans['merchant_name'].replace('None', None)

    # create a column combined_merchant where we take the any merchant name : Cleo, or plaid, or counterparty 
    df_trans['merchant_name_combined'] = df_trans['merchant_name'].combine_first(df_trans['merchant_name_plaid'])
    # if counterparty_type is merchant , user counterparty_name
    #df_trans['merchant_name_combined'] = df_trans['merchant_name_combined'].combine_first(df_trans['counterparty_name'])
    df_trans['merchant_name_combined'] = df_trans['merchant_name_combined'].combine_first(df_trans.apply(lambda row: row['counterparty_name'] if row['counterparty_type']=='merchant' else None, axis=1))
    # Remove data without merchant name for training data
    df_trans = df_trans[(~df_trans['merchant_name_combined'].isnull()) & ~df_trans['merchant_name_combined'].isin(['',' '])][:]
    df_trans['merchant_name_combined_len'] = df_trans['merchant_name_combined'].apply(lambda x: len(x))
    df_trans = df_trans[df_trans['merchant_name_combined_len']>=1]
    df_trans.drop('merchant_name_combined_len', axis=1, inplace=True)

    # Coalescing descriptions 
    # if original_description_plaid is empty use description
    df_trans['description_combined'] = df_trans['original_description_plaid'].combine_first(df_trans['description'])
    df_trans['len_description'] = df_trans['description_combined'].apply(lambda x: len(x))
    df_trans = df_trans[df_trans['len_description'] >=2]
    df_trans.drop('len_description', axis=1, inplace=True)
    ##
    #replace 'other' with ''
    df_trans['payment_channel_processed'] = df_trans['payment_channel'].apply(lambda x: 'None' if x == 'other' else x)
    # do some light processing to make strings shorter
    df_trans['description_combined_processed'] =  df_trans['description_combined'].apply(lambda x: re.sub('\\\\+','\\\\',x))
    df_trans['description_combined_processed'] =  df_trans['description_combined_processed'].apply(lambda x: re.sub(r'\d{4,}', ' ', x))
    df_trans['description_combined_processed'] =  df_trans['description_combined_processed'].apply(lambda x: re.sub(r'\d{4,}', ' ', x))
    df_trans['description_combined_processed'] =  df_trans['description_combined_processed'].apply(lambda x: re.sub(r'(.)\1{4,}', ' ', x))
    df_trans['description_combined_processed'] =  df_trans['description_combined_processed'].apply(lambda x: re.sub(' +',' ',x))
    df_trans.reset_index(drop=True, inplace=True)

    # examples where the description and the merchant name are the same are probably not too informative
    df_trans = df_trans[df_trans['merchant_name_combined']!=df_trans['original_description_plaid']]


    # create some sentences
    df_trans['amount'] = df_trans['amount'].round(1)
    df_trans['str_amount'] = df_trans['amount'].apply(lambda x: str(x))
    df_trans['sentence'] = df_trans['description_combined_processed'] + '. Channel: ' + df_trans['payment_channel_processed'] + '. Amount: ' + df_trans['str_amount']
    df_trans['sentence2'] = df_trans['description_combined_processed'] +'. Type: ' +df_trans['counterparty_type']+'. Channel: ' +\
        df_trans['payment_channel_processed'] + '. Amount: ' + df_trans['str_amount']
    try:
        df_trans.drop('str_amount', axis=1, inplace=True)
    except:
        pass
    df_trans['len_sentence'] = df_trans['sentence'].apply(lambda x: len(x))
    df_trans['num_words'] = df_trans['sentence'].apply(lambda x: len(x.split(' ')))

    # There are some cases where Chime is the merchant but it isn't mentioned in the description, so remove these. I am sure there are more like these, we would need to see
    df_trans['Chime in descr'] = df_trans['original_description_plaid'].apply(lambda x: 'Chime' in x)

    df_trans = df_trans[(df_trans['Chime in descr'] & (df_trans['merchant_name_combined']=='Chime')) | (df_trans['merchant_name_combined']!='Chime')]
    df_trans.drop('Chime in descr', axis=1, inplace=True)
    df_trans.reset_index(drop=True, inplace=True)

    df_trans_no_merchant = df_trans[(df_trans['merchant_name_combined'].isnull()) | df_trans['merchant_name_combined'].isin(['',' '])]

    # examples where the description and the merchant name are the same are probably not too informative
    df_trans = df_trans[df_trans['merchant_name_combined']!=df_trans['original_description_plaid']]
    print(df_trans.shape)
    #df_trans_cln3['merchant_name_combined'].value_counts()[:-40]
    df_trans = df_trans.drop_duplicates(subset=['original_description_plaid','merchant_name_combined'])
    df_trans.shape
    #pre-processing about halves the volume of data


    columns_to_keep = ['transaction_id','corrected_made_on','amount','description_combined','merchant_name_combined','description_combined_processed','sentence','sentence2','payment_channel','currency_code','original_description_plaid','description']
    nunique_merchants = df_trans['merchant_name_combined'].nunique()
    print(f"There are {nunique_merchants} merchants in the dataset")
    df_trans['merchant_name_combined'].value_counts(dropna=False)
    path_file_processed_p = path_file_processed+'.parquet'
    print(path_file_processed_p)
    wr.s3.to_parquet(
        df=df_trans[columns_to_keep],
        path=path_file_processed,
        dataset=True #,partition_cols=["merchant_name_combined"]
    )

    print(f"Finsihed writing file {path_file_processed_p}")


# %%



