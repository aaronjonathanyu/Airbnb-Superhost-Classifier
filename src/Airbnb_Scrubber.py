import sys, os
import numpy as np
import pandas as pd
import string

#langdetect
from langdetect import detect

#NLTK
import string
import nltk
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

#####################################################################
#  Step 1: create 'en_review_filter'
#####################################################################
def en_review_filter(df_reviews):
    '''
        Input : Pass in a dataframe of reviews
        Output: returns a new df_reviews filtered for english reviews only
        based on the 'comments' column
    '''
    #create test__df from df_reviews
    test_df = df_reviews
    #clean test_df.comments of everything except for letters a-zA-Z and replace it with ''
    test_df.comments = test_df.comments.str.replace(r'[^ a-zA-Z]','')
    #strip all white spaces
    test_df.comments = test_df.comments.str.strip()
    #fill all nans with ''
    test_df.comments.fillna('', inplace=True)
    #create mask that shows me all comments excluding blank reviews ''
    mask=test_df['comments']!= ''
    #apply mask to test_df and utilize detect to detect the language of reviews, make a new column called Languagereview that shows language of review.
    test_df['Languagereview'] = test_df['comments'][mask].apply(detect)
    #create new column Languagereview for df_reviews and apply test_df.Languagereview to it.
    df_reviews['Languagereview'] = test_df.Languagereview
    #update df_reviews with only 'en' reviews
    df_reviews = df_reviews[df_reviews['Languagereview']== 'en']
    return None

#####################################################################
#  Step 2: create 'en_review_scrubber'
#####################################################################

def en_review_scrubber(df_reviews):
    '''
        Input : Pass in a dataframe of reviews
        Output: Returns a new df_reviews that has been scrubbed for english words only
        based on the 'comments' column
    '''
    #clean reviews of the word "super host" and "superhost" and replace with ""
    #prevent leakage*
    df_reviews.comments.replace(to_replace="super host",value="",inplace= True, regex=True)
    df_reviews.comments.replace(to_replace="superhost*",value="",inplace= True, regex=True)
    # extracting english stopwords from nltk library
    sw = stopwords.words('english')
    english_sw = "|".join(sw)
    #Create a combined_alphabet/chinese/japanese/korean stopword dict
    combined_alphabet = "a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z"
    chinese_sw='得|可|些|么|用|则|致|而|咱|虽|即|另|怎|看|儿|几|那|嗡|何|了|她|往|或|它|嘛|曾|在|哟|各|起|从|既|之|与|尔|又|的|好|给|今|后|如|哪\
                |但|再|随|吧|人|别|向|当|该|趁|您|打|若|为|并|被|啦|哇|据|有|我|谁|由|们|让|每|同|最|上|所|却|跟|也|个|且|一|不|距|你|拿|此|去|诸\
                |其|着|来|到|以|乃|他|沿|把|自|至|啥|仍|是|已|小|凡|无|还|只|凭|于|靠|比|使|得|可|些|么|性|这|鍵|总'
    japenese_sw='で|い|か|と|え|し|よ|を|り|達|れ|の|た|方|ま|こ|も|が|だ|女|お|ど|私|々|あ|す|は|な|彼|そ|ち|ら|貴|に|ん|ｷ|ｯ|ﾁ|ﾝ|や|て|б'
    korean_sw='임|술|버|직|훨|보|졸|허|틈|더|겨|조|하|여|아|리|진|툭|곳|쪽|할|차|들|왜|렁|번|당|답|한|큼|희|&|상|었|둘|쩔|컨|육|가|히|남|후|흥|좋|렵|안|르\
                |저|합|겸|경|둥|등|방|쨋|적|영|견|미|누|때|위|좍|려|수|연|쿠|일|함|q|떤|약|갖|퉤|과|봐|른|삼|몰|예|섯|불|길|공|이|총|지|곧|은|깐|않|라|윗\
                |림|금|향|야|든|걸|모|륙|소|주|해|령|슨|장|비|펄|게|겠|메|두|그|망|키|붕|신|시|까|네|준|물|스|=|익|또|듯|우|관|입|으|혹|낼|||참|팍|잇|럼\
                |알|거|틀|교|못|실|련|착|무|어|꿔|을|각|국|째|논|형|유|줄|에|댕|될|았|의|구|사|부|머|같|즈|팔|딩|퍽|로|닭|통|문|름|점|켠|결|다|오|휴|김|를\
                |류|걱|서|양|악|밖|군|운|생|종|써|슷|씬|선|달|계|쾅|도|람|된|따|낫|개|동|편|넷|칠|응|나|언|젠|좀|추|외|흐|콸|월|끼|릎|뿐|목|는|중|천|대|뚝\
                |렇|냐|정|설|근|없|얼|휘|마|혼|옆|것|기|용|년|삐|앞|쳇|데|바|전|됏|찌|많|간|힘|록|쓰|반|식|자|놀|초|고|즉|윙|했|럴|런|너|탕|론|랏|께|되|호\
                |겁|o|떡|엉|며|집|엇|꽈|막|단|래|분|쉿|디|헐|룩|짓|앗|뒤|항|딱|토|곱|인|및|본|울|매|드|말|짜|쿵|체|므|제|셋|있|떠|느|솨|헉|작|였|욱|첫|러\
                |치|심|터|면|떻|곤|홉|덟|습|타|와|끙|꾸|요|만|득|잠|몇|뜌|씰|꺼|듭|댜|듦|홴|ㅠ|흑'
    #create new df_reviews with only english alphabet & english stop words while removing reviews with chinese, korean, or japanese stopwords
    df_reviews=df_reviews[df_reviews['comments'].str.contains(combined_alphabet, na=False)==True]
    df_reviews=df_reviews[df_reviews['comments'].str.contains(english_sw, na=False)==True]
    df_reviews=df_reviews[df_reviews['comments'].str.contains(chinese_sw, na=False)!=True]
    df_reviews=df_reviews[df_reviews['comments'].str.contains(japenese_sw, na=False)!=True]
    df_reviews=df_reviews[df_reviews['comments'].str.contains(korean_sw, na=False)!=True]
    return None


#####################################################################
#  Step 3: create 'review_consolidator'
#####################################################################

def review_consolidator(df_reviews):
    '''
        Input : Pass in a dataframe of reviews
        Output: Returns a new df_reviews that groups together all reviews from the comments column per listing
    '''
    #create a df_reviews where we group all comments by listing_id
    df_reviews = df_reviews.groupby(['listing_id'])['comments'].apply(','.join).reset_index()
    return None


#####################################################################
#  Step 4: output df_reviews as json file 'review_tojson'
#####################################################################

def review_tojson(df_reviews):
    '''
        Input : Pass in a dataframe of reviews
        Output: df_reviews to a json file
    '''
    #create a df_reviews.json file in current directory
    return df_reviews.to_json('df_reviews.json',orient='split')


#####################################################################
#  Step 5: create 'listing_cleaner'
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
#  Step 6: output df_listings as json file 'listings_tojson'
#####################################################################

def df_listings_tojson(df_listings):
    '''
        Input : Pass in a dataframe of listings
        Output: df_listings to a json file
    '''
    #create a df_listings.json file in current directory
    return df_listings.to_json('df_listings.json',orient='split')


#####################################################################
#  Step 7: Combine df_listings and df_reviews "combine_df"
#####################################################################

def combined_df():
    '''
        Input : Pass in reviews and listing json file
        Output: A combined file of reviews and listings called df_combined
    '''
    #open df_reviews and df_reviews from current directory
    df_reviews = pd.read_json('json_files/df_reviews.json', orient='split')
    df_listings = pd.read_json('json_files/df_listings.json', orient='split')
    #merge df_reviews and df_listings on 'id and listing id'
    df_combined=pd.merge(df_listings, df_reviews, left_on='id', right_on='listing_id', how='left')
    return df_combined


#####################################################################
#  Step 8: clean df_combined of unecessary things "combined_scrubber"
#####################################################################

def combined_scrubber(df_combined):
    '''
        Input : Pass in df_combined
        Output: df_combined scrubbed for only necessary information
    '''
    #fill all nan values in host_is_superhost column with "f"
    df_combined['host_is_superhost'].fillna('f', inplace = True)
    #replace all "t" and "f" with boolean value of True and False
    df_combined.replace({'host_is_superhost': {'t': True, 'f': False}},inplace=True)
    #update df_combined to only include listings with reviews
    df_combined=df_combined[df_combined.comments.isnull()==False]
    #fill nans in "description" column with the string "missing description"
    df_combined['description'].fillna('missing description',inplace=True)
    #Create a new column named "Languagereview_description' that lets me know what language each listings's description is
    description_mask=df_combined['description']!=''
    df_combined['Languagereview_description'] = df_combined['description'][description_mask].apply(detect)
    #Create a new column named "Languagereview_comments' that lets me know what language each listings's comments/reviews is (2nd filter)
    comments_mask=df_combined['comments']!=''
    df_combined['Languagereview_comments'] = df_combined['comments'][comments_mask].apply(detect)
    #drop unecessary columns
    df_combined.drop(['scrape_id','listing_url','last_scraped','name','summary','space',\
                  'neighborhood_overview','notes','access','interaction','house_rules',\
                  'thumbnail_url','medium_url','picture_url','xl_picture_url','host_url',\
                  'host_name','host_location','host_about','host_thumbnail_url',\
                  'host_picture_url','host_neighbourhood','host_total_listings_count','street',\
                  'neighbourhood_cleansed','neighbourhood_group_cleansed','state','market',\
                  'smart_location','country_code','country','square_feet','extra_people',\
                  'calendar_last_scraped','requires_license','license','jurisdiction_names',\
                  'require_guest_profile_picture','require_guest_phone_verification',\
                  'calculated_host_listings_count','listing_id', 'experiences_offered', 'host_acceptance_rate'\
                 ],axis=1,inplace=True)
    #reset index for df_combined
    df_combined.reset_index(drop=True, inplace=True)
    return None


#####################################################################
#  Step 9: output df_combined as json file 'combined_tojson'
#####################################################################

def df_combined_tojson(df_combined):
    '''
        Input : Pass in a dataframe df_combined
        Output: df_combined to a json file
    '''
    #create a df_combined.json file in current directory
    return df_combined.to_json('df_combined.json',orient='split')


#####################################################################
#  Step 10: read in df_combined for "nlp"
#####################################################################

def df_combined_tojson(df_combined_file):
    '''
        Input : Read in df_combined
        Output: df_combined to a json file
    '''
    df_combined = pd.read_json('final_files/df_combined.json', orient='split')



