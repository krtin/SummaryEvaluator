import numpy as np
import pandas as pd
import os
import pickle as pkl
import itertools
import scipy.stats

toutanova_corpus_raw = "data/toutanova_corpus"
#disc_model = "modeltest"
#disc_model = "weightedmodel"
disc_model = "attnmodel_tout"
list_of_files = [f for f in os.listdir(toutanova_corpus_raw) if os.path.isfile(os.path.join(toutanova_corpus_raw, f))]
def getscores(humanrating):
    if(humanrating==6):
        return 3, 3
    elif(humanrating==7):
        return 3, 2
    elif(humanrating==9):
        return 3, 1
    elif(humanrating==11):
        return 2, 3
    elif(humanrating==12):
        return 2, 2
    elif(humanrating==14):
        return 2, 1
    elif(humanrating==21):
        return 1, 3
    elif(humanrating==22):
        return 1, 2
    elif(humanrating==24):
        return 1, 3
    else:
        raise Exception("No value match for score mapping: %d"%humanrating)

data_sets = []
for filename in list_of_files:
    if("test" not in filename):
        continue

    filepath = os.path.join(toutanova_corpus_raw, filename)
    datatype = filename.split("_")
    if(len(datatype)<2):
        datatype = "test"
        continue
    else:
        datatype = datatype[1].split(".")[1]

    print(filename)
    error = 0
    count= 0
    data_algo=[]
    with open(filepath, 'r') as f:
        data = f.read().split('\n')
        for row in data:
            if(row==''):
                continue
            row = row.split('|||')
            if(len(row)!=2):
                print("Compression info does not exists")
                #print(row[0])
                error+=1
                continue

            sourceinfo = row[0]
            compressioninfo = row[1]

            [sourceid, domain, source] = sourceinfo.split('\t')
            #print(sourceid)
            #print(domain)
            #print(source)
            compressioninfo = compressioninfo.split('\t')
            if(len(compressioninfo)<3):
                raise Exception("Error processing compression info")
            summary = compressioninfo[0]
            judgeid = compressioninfo[1]
            noof_ratings = int(compressioninfo[2])

            #print(summary)
            #print(judgeid)
            #print(noof_ratings)

            count += 1
            humanratings = compressioninfo[3:]

            g_scores = []
            m_scores = []
            for idx, humanrating in enumerate(humanratings):
                if(idx%2==0):
                    #get grammaticality and meaning score from human ratings
                    g_score, m_score = getscores(int(humanrating))
                    g_scores.append(g_score)
                    m_scores.append(m_score)
            g_mean = np.mean(g_scores)
            m_mean = np.mean(m_scores)
            data_algo.append({"sourceid":sourceid, "g_mean_"+datatype:g_mean, "m_mean_"+datatype:m_mean, "noof_ratings_"+datatype:noof_ratings})

    data_algo = pd.DataFrame(data_algo)
    filepath = os.path.join(os.path.join(os.path.join(disc_model, "decode_"+datatype), "probs"),"probs.pkl")
    [disc_probs, prob_len] = pkl.load(open(filepath, 'rb'))

    if(len(disc_probs)!=len(data_algo)):
        raise Exception("The length of corpus and generated probs don't match")
    data_algo["disc_"+datatype] = disc_probs
    data_algo.drop_duplicates('sourceid', keep=False, inplace=True)

    print("Repeated sourceids %d"%(len(data_algo['sourceid']) - len(data_algo['sourceid'].unique())))

    #print(data_algo.groupby('sourceid', axis=0)['sourceid'].count())

    data_sets.append(data_algo)
    #print(count)


count = 0
complete_data=[]
for df in data_sets:
    if(count==0):
        complete_data = df
        count+=1
        continue
    complete_data = pd.merge(complete_data, df, on="sourceid", how='inner')

    count+=1

#print(first_df)
pkl.dump([complete_data], open("data/correlation/alldata.pkl", 'wb'))

#test = np.array(list(complete_data['disc_ilp']))
#print(len(test[:, 0]))
#raise Exception('Test')

data_set_names = ["ilp", "t3", "namas", "seq2seq"]
#data_set_names = ["ilp", "namas"]
combs = itertools.combinations(data_set_names, 2)

for comb in list(combs):
    model1 = comb[0]
    model2 = comb[1]
    mean_human = complete_data['m_mean_'+model1]-complete_data['m_mean_'+model2]
    gram_human = complete_data['g_mean_'+model1]-complete_data['g_mean_'+model2]
    score_disc = -(-complete_data['disc_'+model2])
    #model1_scores = np.array(list(complete_data['disc_'+model1]))[:, 1]
    #model2_scores = np.array(list(complete_data['disc_'+model2]))[:, 1]
    #score_disc = model1_scores - model2_scores
    #print(mean_human)
    #print(complete_data['disc_'+model1])
    #print(gram_human)
    corr_m, pvalue_m = scipy.stats.pearsonr(list(mean_human), list(score_disc))
    corr_g, pvalue_g = scipy.stats.pearsonr(list(gram_human), list(score_disc))

    print("Meaning Correlation and pvalue", corr_m, pvalue_m, comb[0], comb[1])
    print("Grammaticality Correlation and pvalue", corr_g, pvalue_g, comb[0], comb[1])
    print()
