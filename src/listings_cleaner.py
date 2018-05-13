import sys, os
import numpy as np
import pandas as pd
#langdetect
from langdetect import detect
#NLTK
from nltk.corpus import stopwords

#####################################################################
                    ### Listings Scrubber ###
#####################################################################



#####################################################################
#  Step 1: create 'listing_cleaner'
#####################################################################

def listing_cleaner(df_listings):
    '''
        Input : Pass in a dataframe of listings
        Output: Returns a new df_listings that has been all respective zipcodes filled
    '''
    #convert df_listings.zipcode to integer type and fill in all nans with 0's
    df_listings.zipcode = pd.to_numeric(df_listings.zipcode, errors='coerce').fillna(0).astype(np.int64)
    #create a test df that only gives me all zipcodes that is not zero, used this to create a dict.
    test=df_listings[df_listings.zipcode != 0]
    #create a dictionary named test1 of neighbourhoods and their respective zipcodes to be used to fill in zipcodes with 0's
    nbhd_zip_dict=dict(zip(test.neighbourhood.values,test.zipcode.values))
    #create df_nbhd from the nbhd_zip_dict dictionary, only includes neighbourhood & zipcode
    df_nbhd = pd.DataFrame.from_dict(nbhd_zip_dict,'index')
    df_nbhd.reset_index(inplace=True)
    df_nbhd.columns = ['neighbourhood','zipcode']
    #merge df_listings and df_nbhd on neighbourhood to create df_combined
    df_combined = pd.merge(df_listings, df_nbhd, how='left', on='neighbourhood')
    #fill in zipcodes to df_combined from df_combined.zipcode_x
    df_combined['zipcode_fill'] = df_combined.zipcode_x
    #create a mask where we fill in all missing zipcodes with the respective zipcode based on neighbourhood
    zero_mask = df_combined.zipcode_fill == 0
    df_combined.zipcode_fill.loc[zero_mask] = df_combined.zipcode_y[zero_mask]
    #make df_listings.zipcode column the same as the df_combined column, while dropping all nan's with 10001 and converting all floats to int
    df_listings.zipcode= df_combined.zipcode_fill.fillna(10001).astype(np.int64)
    return None


#####################################################################
#  Step 2: output df_listings as json file 'listings_tojson'
#####################################################################

def df_listings_tojson(df_listings):
    '''
        Input : Pass in a dataframe of listings
        Output: df_listings to a json file
    '''
    #create a df_listings.json file in current directory
    return df_listings.to_json('df_listings.json',orient='split')
