# Load required libraries
import matplotlib as plt
plt.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.externals import joblib
import pandas as pd
import numpy as np

# Set figure display options
sns.set(context='notebook', style='darkgrid')
sns.set(font_scale=1.4)

def construct_graph(scaled_meta_features, coefficients, top_project_std):
    # List of meta features
    meta_feat=['sentiment','ADJ','ADP','ADV','CCONJ','DET','INTJ','NOUN','NUM',
               'PART','PRON','PROPN','PUNCT','SPACE','SYM','VERB','X',
               'avg_word_sent','num_sent','num_word']

    # Compute feature importances of the meta features
    feature_ranks = pd.Series(coefficients[:len(meta_feat)],index=meta_feat)
    
    # List of meta features that were most predictive of funded projects
    pred_feat=['sentiment','SYM','DET','PUNCT','ADJ','VERB']

    # Transform the standardized feature vector into a Series
    feature_vector_std = pd.Series(scaled_meta_features.ravel(), index=meta_feat)

    # Compute the weighted score of the meta features of the user's project
    user_nar_score = np.multiply(feature_vector_std[pred_feat],
        feature_ranks[pred_feat])

    # Compute the weighted score of the meta features of the average top project
    suc_nar_score = np.multiply(top_project_std[pred_feat],
        feature_ranks[pred_feat])

    # Combine the weighted score into a single DataFrame
    messy = pd.DataFrame([user_nar_score, suc_nar_score], 
            index=['Your Narrative', 'Successful Narratives']).T.reset_index()

    # Transform the combined data into tidy format
    tidy = pd.melt(messy,id_vars='index', 
           value_vars=['Your Narrative', 'Successful Narratives'],var_name=' ')

    fig = sns.factorplot(data=tidy, y='index', x='value', hue=' ', kind='bar',
    size=5, aspect=1.5, palette='Set1', legend_out=False).set(xlabel='score',
    ylabel='',xticks=[])

    # Re-label the y-axis and reposition the legend
    labels = ['Sentiment','Symbols','Determiners','Punctuation','Adjectives','Verbs']
    plt.yticks(np.arange(len(pred_feat)), labels)
    fig.ax.legend(loc='lower right');

    plt.savefig(
        '/home/wescrues/Insight/flapp/static/images/figure.png',
        dpi=300,
        bbox_inches='tight'
    );
