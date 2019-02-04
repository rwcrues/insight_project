# Load required libraries
from flask import Flask, render_template, request
from sklearn.externals import joblib
from math import floor
from scipy import sparse
import pandas as pd
import numpy as np
from random import random
from sklearn.preprocessing import StandardScaler
import feat_eng
import fig_gen

#Load model trained on meta features and n-grams
clf = joblib.load('trained_classifier.pkl')
vectorizer = joblib.load('tfidf_unibi_250.pkl')

#successful complaints meta features
suc_comp=joblib.load('success_meta_vector.pkl')

#load pre-trained scaler
scaler=joblib.load('trained_scaler.pkl')

#initialize flask app
app=Flask(__name__, static_url_path='/static')

@app.route('/')
@app.route('/index')
def index():
     return render_template('index.html')

@app.route('/output')
def narpot_output():

    #get the narrative from the input and store it
    narrative=request.args.get('narrative')
    
    #randomized links to prevent caching
    img_link = '../static/images/figure.png?q=' + str(random())
    css_link = '../static/css/custom.css?q=' + str(random())

    #take the narrative text and analyze it so that we can get a prediction
    # first, get sentiment
    senti = feat_eng.sentiment_analyzer_scores(narrative)
    # Take input text and get POS tags
    pos = feat_eng.postag(narrative)
    # Take input text and get summary statistics about length
    length = feat_eng.sent_word_tok(narrative)

    #create pos table so all variables are present
    pos_df=pd.DataFrame()
    pos_df["ADJ"]=0; pos_df["ADP"]=0; pos_df["ADV"]=0; pos_df["CCONJ"]=0; 
    pos_df["DET"]=0; pos_df["INTJ"]=0; pos_df["NOUN"]=0; pos_df["NUM"]=0; 
    pos_df["PART"]=0; pos_df["PRON"]=0; pos_df["PROPN"]=0; pos_df["PUNCT"]=0; 
    pos_df["SPACE"]=0; pos_df["SYM"]=0; pos_df["VERB"]=0; pos_df["X"]=0

    #change these all to a pandas data frame and concatenate them
    senti_df=pd.DataFrame(pd.Series(senti))
    senti_df.columns=['senti']
    pos_df_data=pd.DataFrame(pos,index=[0])
    pos_fin=pos_df.append(pos_df_data)
    pos_fin=pos_fin.fillna(0)
    length_df=pd.DataFrame(length,index=[0])
    #concatenate these to form meta-feature vector
    meta_feat=pd.merge(senti_df,pos_fin,left_index=True, right_index=True)
    meta_feat=pd.merge(meta_feat,length_df,left_index=True, right_index=True)

    #now, get bag-of-words
    clean_text=feat_eng.standardize_text(narrative)

    #need to get vectorizer from sci-kit learn to create bag-of-words
    # create the transform
    X_ngrams = vectorizer.transform([clean_text])

    #scale the meta-features
    X_meta = scaler.transform(meta_feat)

    #generate graph of recommendations here
    coeff=clf.coef_.T.ravel()
    meta_comparison=fig_gen.construct_graph(X_meta,coeff,suc_comp)

    # Combine the meta features with the n-grams
    X_meta_features = sparse.csr_matrix(X_meta)
    X_full = sparse.hstack([X_meta_features, X_ngrams])
    
    # Compute the probability that the complaint won't be closed with relief 
    prob = floor(100 * clf.predict_proba(X_full)[0, 1])
 
    #generate the output
    return render_template('output.html',the_result=prob,img_hash=img_link,
            css_hash=css_link)

@app.route('/')
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)




