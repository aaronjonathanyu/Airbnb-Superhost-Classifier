import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from langdetect import detect

#####################################################################
#  Step 1: create 'Filter_Reviews'
#####################################################################
def review_filter(df_reviews):
    '''
        Input : pass in a dataframe of reviews
        Output: returns a new df_reviews filtered for english reviews only
        based on the 'comments' column
    '''
    #Replace the number 3 with 'three' for review #690564
    df_reviews.iloc[[690564],[5]]='three'
    #Replace the heart emoji with 'heart' for review #872002
    df_reviews.iloc[[872002],[5]]='heart
    #Replace foreign language with character to be able to identify lang for review #567534
    df_reviews.iloc[[567534],[5]]='의'
    # extracting english stopwords from nltk library
    sw = stopwords.words('english')
    english_sw = "|".join(sw)
    #Create a combined_alphabet/chinese/japanese/korean stopword dict
    combined_alphabet = "a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z"
    chinese_sw='得|可|些|么|用|则|致|而|咱|虽|即|另|怎|看|儿|几|那|嗡|何|了|她|往|或|它|嘛|曾|在|哟|各|起|从|既|之|与|尔|又|的|好|给|今|后|如|哪|但|再|随|吧|人|别|向|当|该|趁|您|打|若|为|并|被|啦|哇|据|有|我|谁|由|们|让|每|同|最|上|所|却|跟|也|个|且|一|不|距|你|拿|此|去|诸|其|着|来|到|以|乃|他|沿|把|自|至|啥|仍|是|已|小|凡|无|还|只|凭|于|靠|比|使|得|可|些|么'
    japenese_sw='で|い|か|と|え|し|よ|を|り|達|れ|の|た|方|ま|我|こ|も|が|だ|女|お|ど|私|々|あ|何|す|は|人|な|彼|そ|ち|ら|貴|に|ん'
    korean_sw='임|술|버|직|훨|보|졸|허|틈|더|겨|조|하|여|아|리|진|툭|곳|쪽|할|차|들|왜|렁|번|당|답|한|큼|희|&|상|었|둘|쩔|컨|육|가|히|남|후|흥|좋|렵|안|르|저|합|겸|경|둥|등|방|쨋|적|영|견|미|누|때|위|좍|려|수|연|쿠|일|함|q|떤|약|갖|퉤|과|봐|른|삼|몰|예|섯|불|길|공|이|총|지|곧|은|깐|않|라|윗|림|금|향|야|든|걸|모|륙|소|주|해|령|슨|장|비|펄|게|겠|메|두|그|망|키|붕|신|시|까|네|준|물|스|=|익|또|듯|우|관|입|으|혹|낼|||참|팍|잇|럼|알|거|틀|교|못|실|련|착|무|어|꿔|을|각|국|째|논|형|유|줄|에|댕|될|았|의|구|사|부|머|니|같|즈|팔|딩|퍽|로|닭|통|문|름|점|켠|결|다|오|휴|김|를|류|걱|서|양|악|밖|군|운|생|종|써|슷|씬|선|달|계|쾅|도|람|된|따|낫|개|동|편|넷|칠|응|나|언|젠|좀|추|외|흐|콸|월|끼|릎|뿐|목|는|중|천|대|뚝|렇|냐|정|설|근|없|얼|휘|마|혼|옆|것|기|용|년|삐|앞|쳇|데|바|전|됏|찌|많|간|힘|록|쓰|반|식|자|놀|초|고|즉|윙|했|럴|런|너|탕|론|랏|께|되|호|겁|o|떡|엉|며|집|엇|꽈|막|단|래|분|쉿|디|헐|룩|짓|앗|뒤|항|딱|토|곱|인|및|본|울|매|드|말|짜|쿵|체|므|제|셋|있|떠|느|솨|음|헉|작|였|욱|첫|러|치|심|터|면|떻|곤|홉|덟|습|타|와|끙|꾸|요|만|득|잠|몇'
    #create new df_reviews with only english alphabet & english stop words while removing reviews with chinese, korean, or japanese stopwords
    df_reviews=df_reviews[df_reviews['comments'].str.contains(combined_alphabet, na=False)==True]
    df_reviews=df_reviews[df_reviews['comments'].str.contains(english_sw, na=False)==True]
    df_reviews=df_reviews[df_reviews['comments'].str.contains(chinese_sw, na=False)!=True]
    df_reviews=df_reviews[df_reviews['comments'].str.contains(japenese_sw, na=False)!=True]
    df_reviews=df_reviews[df_reviews['comments'].str.contains(korean_sw, na=False)!=True]
    df_reviews=df_reviews[df_reviews['comments'].str.contains('ｷｯﾁﾝや鍵|힘|한|이을|들|었|음|ㅠ|뜌|씰|꺼|예|여|듭|니|댜|다|듦|들|었|지|요|만|홴|总|흑|편|性|这|是|て|б|靠', na=False)!=True]
    #detect language using langdetect package
    #first create mask so we can filter out reviews that do not have any thing in their fields
    mask=df_reviews['comments']!=''
    #create a new column named "Languagereview' that lets me know what each row's comment section language is
    #need this to only keep english comments/reviews
    df_reviews['Languagereview'] = df_reviews['comments'][mask].apply(detect)
    #df_reviews is filtered only english reviews based on the Languagereview column
    df_reviews = df_reviews[df_reviews['Languagereview']== 'en']
    return df_reviews


#####################################################################
#  Step 2: create 'Superhost_Not_Superhost'
#####################################################################
def scrub_fraud_no_fraud(df):
    '''
        Input : pass in a dataframe
        Output: returns a series of Booleans based on 'acct_type'
    '''
    df['fraud_no_fraud'] = df['acct_type'].astype(str).str[:5] == 'fraud'
    return df['fraud_no_fraud']


#####################################################################
#  Step 2: create 'fraud_no_fraud'
#####################################################################
def scrub_fraud_no_fraud(df):
    '''
        Input : pass in a dataframe
        Output: returns a series of Booleans based on 'acct_type'
    '''
    df['fraud_no_fraud'] = df['acct_type'].astype(str).str[:5] == 'fraud'
    return df['fraud_no_fraud']
