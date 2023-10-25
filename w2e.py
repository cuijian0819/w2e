import math
import random
import pandas as pd
import numpy as np
from tqdm import tqdm 
from collections import Counter
from datetime import datetime, timedelta

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from scipy.spatial import distance

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

import spacy
nlp = spacy.load('en_core_web_sm', disable=['textcat', 'parser', 'ner'])

from pdb import set_trace

# Load pre-defined word list
with open('data/D_common.txt', 'r') as f:
    D_common = set([line.strip() for line in f])
        
with open('data/D_tech.txt', 'r') as f:
    D_tech = set([line.strip() for line in f])

with open('data/D_whitelist.txt', 'r') as f:
    D_whitelist = set([line.strip() for line in f])

def preprocess_w2e(text):
    # text = preprocess_text(text)
    result = []
    for token in nlp(text, disable=['textcat', 'parser', 'ner']):
        if (not token.is_stop) and token.pos_ in ('PROPN', 'NOUN', 'PRON'):
            token.lemma_.lower()
            result.append(token.text)

    result = list(set(result))
    return result

def count_tokens(token_list):
    word = [t for tl in token_list for t in tl]
    counter = Counter(word)
        
    return counter, set(list(counter.keys()))

def get_significat_words(word_list, n, current_c):
    z = 1.645
    word_list_ = []
    for w in word_list:
        f_w = current_c[w]
        p_w = f_w/n
        if p_w >= z*math.sqrt(p_w*(1-p_w)/n): 
            word_list_.append(w)
    return word_list_

def get_new_words(C, K, n, current_c):
    new_words = set(K - C - D_common.union(D_tech))
    new_words_ = get_significat_words(new_words, n, current_c)
    print(f"new words before: {len(new_words)}; new words after: {len(new_words_)}; ")
    return new_words_

# Load data
tweet_df = pd.read_excel('dummy_tweets.xlsx')
# tweet_df = tweet_df.drop_duplicates(subset=['text'])
tweet_df['created_at'] = pd.to_datetime(tweet_df['created_at'])


start_date = datetime(2022, 11, 5)
end_date = datetime(2022, 11, 30)
one_day = timedelta(days=1)

# Initialize the current date
current_date = start_date
concatenated_documents = []
start_list, end_list = [], []
# Iterate through the dates in November
while current_date <= end_date:
   
    # last two days data in the list
    tweets_past_list = []
    tweets_past_list.append(tweet_df[(tweet_df['created_at'] >=  current_date - timedelta(days=3)) & (tweet_df['created_at'] <= current_date - timedelta(days=2))].reset_index(drop=True))
    tweets_past_list.append(tweet_df[(tweet_df['created_at'] >=  current_date - timedelta(days=2)) & (tweet_df['created_at'] <= current_date - timedelta(days=1))].reset_index(drop=True))
    tweets_past_list.append(tweet_df[(tweet_df['created_at'] >=  current_date - timedelta(days=1)) & (tweet_df['created_at'] <= current_date)].reset_index(drop=True))
    tweets_cur = tweet_df.loc[(tweet_df['created_at']>= current_date) & (tweet_df['created_at'] <= current_date + timedelta(days=1))].reset_index(drop=True)

    tweets_past_list_p = [tl['text'].apply(lambda x: preprocess_w2e(x)) for tl in tweets_past_list]
    tweets_cur['text_p'] = tweets_cur['text'].apply(lambda x: preprocess_w2e(x))

    '''
    New Words
    '''
    tokens_last  = tweets_past_list_p[-1].tolist()
    tokens_cur = tweets_cur.text_p.tolist()

    # C: last, t-1 ~ t
    # K: current, t
    last_c, C = count_tokens(tokens_last) # return (token, count)
    current_c, K = count_tokens(tokens_cur)
    n = len(tokens_last)+1

    new_words = get_new_words(C, K, n, current_c)

    '''
    Re-emerging keywords
    '''
    past_all_df = pd.DataFrame()
    for i, text_p in enumerate(tweets_past_list_p):
        past_c, _ = count_tokens(text_p) 
        past_df = pd.DataFrame.from_dict(past_c, orient='index').reset_index()
        past_df = past_df.rename(columns={'index':'token', 0:'freq'})
        past_df['ts'] = i 
        past_all_df = pd.concat([past_all_df, past_df])


    smoothing_f = 0.4
    reemerge_words = []
    k = len(tweets_past_list_p) # 3
    sf = (smoothing_f * (1 - (1 - smoothing_f)**(2 * k))) / (2 - smoothing_f)

    C_R = C.intersection(D_tech - D_whitelist)
    C_R_ = get_significat_words(C_R, n, current_c)

    past_all_df = past_all_df[past_all_df['token'].isin(C_R_)]

    for token, tmp_df in past_all_df.groupby('token'):
        if tmp_df.shape[0] != 3:
            for ts_ in set([0,1,2]) - set(tmp_df.ts.tolist()):
                new_row = pd.DataFrame.from_dict([{'token':token, 'freq':0, 'ts':ts_}])
                tmp_df = pd.concat([tmp_df, new_row])
            tmp_df.sort_values(by=['ts'], ascending=[True])
        tmp_df['EWMA'] = tmp_df['freq'].ewm(alpha=smoothing_f, adjust=False).mean()
        fw = current_c[token]
        fw_ = tmp_df.iloc[-1]['EWMA']
        tmp_df['sigma'] = (tmp_df['freq'] - tmp_df['EWMA'])**2
        if (fw-fw_)**2 >= 3.8 * tmp_df.sigma.sum()/k * sf:
            reemerge_words.append(token)

    with open('results/reemerge_words.txt', 'w') as f:
        for w in reemerge_words:
            f.write(f"{w}\n")

    with open('results/new_words.txt', 'w') as f:
        for w in new_words:
            f.write(f"{w}\n")

    trigger_words = list(set(reemerge_words).union(set(new_words)))

    mask = tweets_cur['text'].str.contains('|'.join(trigger_words), case=False)

    # Use the mask to filter the DataFrame
    filtered_df = tweets_cur[mask].copy()
    if filtered_df.shape[0] == 0:
        current_date += one_day
        continue

    '''
    Event Detection
    '''
    stop_words = set(stopwords.words('english'))
    filtered_df['tokenized_text'] = filtered_df['text'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.lower() not in stop_words and word.isalpha()]))

    # Calculate the Jaccard distance matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_df['tokenized_text'])
    jaccard_distance = distance.squareform(distance.pdist(tfidf_matrix.todense(), 'jaccard'))

    # Perform hierarchical clustering
    num_clusters = min(10, filtered_df.shape[0])  # Adjust the number of clusters as needed
    clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average')
    cluster_labels = clustering.fit_predict(jaccard_distance)

    # Add cluster labels to the DataFrame
    filtered_df['Cluster'] = cluster_labels

    # Print each cluster
    for cluster_id in range(num_clusters):
        tweets_c = filtered_df[filtered_df['Cluster'] == cluster_id]['text'].tolist()
        tweets_c = random.sample(tweets_c, min(5, len(tweets_c)))
        sampled_documents = [f"T{i}:{text}" for i, text in enumerate(tweets_c)]
        concatenated_documents.append('\n'.join(sampled_documents))
    
    start_list.extend([current_date]*num_clusters)
    current_date += one_day
    end_list.extend([current_date]*num_clusters)


'''
Save result
'''
start_list = [date.strftime('%Y-%m-%d %H:%M:%S') for date in start_list]
end_list = [date.strftime('%Y-%m-%d %H:%M:%S') for date in end_list]

df = pd.DataFrame({
    'start_date': start_list, 
    'endt_date': end_list, 
    'events': concatenated_documents,
    })
df.to_excel('w2e_events.xlsx', index=False)  # Set index=False to exclude row numbers