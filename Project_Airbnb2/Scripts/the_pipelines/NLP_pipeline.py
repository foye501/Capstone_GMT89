import pandas as pd
import spacy
import os
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter

def extract_ner(sentence, spacy_pipeline):
    # Process the input text with the NER component
    if sentence:
        doc = spacy_pipeline(sentence)
        LOCs = []
        # Print the named entities and their labels
        for ent in doc.ents:
            if ent.label_ in ["FAC","LOC","ORG"]:
                # print(ent.text)
                LOCs.append(ent.text)
        return LOCs

import re
# there are some block need to be removed.
def replace_br(locs):
    newlocs=[]
    for loc in locs:
        loc =re.sub('<br /><br /><b',"",loc)
        loc =re.sub('<br />',"",loc)
        loc =re.sub('<br /><br />',"",loc)
        loc =re.sub('<br',"",loc)
        loc =re.sub('/>',"",loc)
        
        newlocs.append(loc)
    return newlocs

# some name need to be unified
def drop_stop(loc):
    new_loc = []
    for word in loc:
        if word not in ['The','the']:
            new_loc.append(word.lower().rstrip('s'))
    return new_loc


# from tqdm import tqdm
# from collections import Counter
# loc_description = []
# ner_count = Counter()
def NLP_pipeline_function(description, spacy_pipeline):
    try:
        ner_list = extract_ner(description, spacy_pipeline)
        ner_list = replace_br(ner_list)
        ner_list = drop_stop(ner_list)
    except:
        ner_list = []
    return ner_list


class NLP_processor():
    def __init__(self):
        self.spacy_pipeline = spacy.load("en_core_web_lg")
        self.sentence_encoder = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')

    def process_airbnb_data(self, df):
        #### 1) NERs
        print('Processing NERs...')
        df_ner = df[['id',"description"]]
        df_ner['description'] = df_ner['description'].fillna("").astype('str')
        df_ner['ner_list'] = [NLP_pipeline_function(i, self.spacy_pipeline) for i in tqdm(df_ner['description'].values)]
        from collections import Counter
        all_count = Counter()
        for l in df_ner['ner_list'].values:
            for e in l:
                all_count.update([e])
        all_count_left = {k:all_count[k] for k in all_count.keys() if all_count[k]>20 and all_count[k]<1000}
        self.NERs_left = list(all_count_left.keys()) #### store the standard NERs used
        df_ner['ner_list_left'] = [[k for k in i if k in all_count_left.keys()] for i in df_ner['ner_list'].values]
        df_exploded = df_ner[['id','ner_list_left']].explode('ner_list_left')
        df_exploded['value'] = 1
        df_exploded = df_exploded.pivot_table(index='id',columns='ner_list_left').droplevel(axis=1, level=0).fillna(0)

        #### 2) Sentence embedding
        print('Processing sentence embedding...')
        embedded_sentences = [{f's{c}':v for c,v in enumerate(self.sentence_encoder.encode(i))} for i in tqdm(df_ner.description.values)]
        embedded_sentences_df = pd.DataFrame(embedded_sentences)

        #### final
        print('Compile all NLP features...')
        final_df = []
        for index,id in enumerate(tqdm(df['id'].values)):
            sub = df_exploded[df_exploded.index==id].reset_index(drop=False)
            if len(sub)==0:
                sub = pd.DataFrame(np.array([0]* len(sub.columns)).reshape(1,-1),columns=sub.columns)
            del sub['id']
            final_df.append({
                'id':id,
                **{a:b for a,b in zip(sub.columns,list(sub.values.flatten()))},
                **{a:b for a,b in zip(embedded_sentences_df.iloc[index,:].index, embedded_sentences_df.iloc[index,:].values)}
            })
        final_df = pd.DataFrame(final_df)
        self.x_names = [i for i in list(final_df.columns) if not i=='id']
        return final_df
    
    def process_new_data(self, description):
        ner_list = NLP_pipeline_function(description, self.spacy_pipeline)
        ner_df = {i:0 for i in self.NERs_left}
        for i in ner_list:
            if i in ner_df.keys():
                ner_df[i]=1

        embedded = self.sentence_encoder.encode(description)
        embedded_df = {f's{i}':v for i,v in enumerate(list(embedded.flatten()))}
        final_df = pd.DataFrame({
            **ner_df,
            **embedded_df
        },index=[0])[self.x_names]
        return final_df
        

        

        
