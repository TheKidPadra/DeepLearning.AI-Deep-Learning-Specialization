# Import your python packages
# author: ismaelali.net

from string import punctuation
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support as score

import pandas as pd


#import warnings
#warnings.filterwarnings("ignore", category=UserWarning, module='')


lemmatzer = WordNetLemmatizer()
stemmer = PorterStemmer()

def lemtz_tokens(tokens, lemmatzer):
    lemmatized = []
    for item in tokens:
        lemmatized.append(lemmatzer.lemmatize(item))
    return lemmatized

def nlkt_tokenize_lemtz(text):
    tokens = nltk.wordpunct_tokenize(text)
    lmtz = lemtz_tokens(tokens, lemmatzer)
    return lmtz



def main():

    # load corpus

    corpus = """
    The students went to their new school today.
    Azad: was the best basketball player in the city.
    Next week, the basketball league will start.
    The school is starting their classes today.
    Best team from UoZ will play with the best team from UoD.
    Teachers were so excited to see their new students.
    Hello Students! Cheered the school principle.
    Did Students start to play basketball from day one?
    """.split("\n")[1:-1]

    print("\n Input Text Corpus: \n")
    print(*corpus, sep="\n")

    print("\n Size of corpus: ", len(corpus),"\n")

    # Pre-Process Text: Punc, Stop Word Removal

    for j, doc in enumerate(corpus):
        result = ''.join(ch for ch in doc if ch not in punctuation)
        corpus[j] = result
        result = ""

    print("\n Text Corpus after punc removal: \n")
    print(*corpus, sep="\n")

    # remove stopwords

    # print(stopwords.words('english'))

    sklearn_tfidf_matrix = TfidfVectorizer(
                                            lowercase=True,
                                            tokenizer= nlkt_tokenize_lemtz,
                                            stop_words=stopwords.words('english')
                                            )
    # Transform Text: TF/IDF
    # transfor corpus_array to corpus_dic

    documents_tokens_dict = {}
    for k, doc in enumerate(corpus):
        documents_tokens_dict[k] = doc

    tf_idf_matrix = sklearn_tfidf_matrix.fit_transform(documents_tokens_dict.values())

    file_IDs = list(documents_tokens_dict.keys())

    print("\n file_IDs : \n " , file_IDs)

    print("\n Terms/Features of TFIDF Matrix: \n", sklearn_tfidf_matrix.get_feature_names_out())

    print("\n TF/IDF matrix: \n")

    for i in range(0, len(documents_tokens_dict)):
        print(file_IDs[i], [round(w,2) for w in tf_idf_matrix.toarray()[i].tolist()])

    # Cluster Text: KMeans

    number_of_clusters = 2

    km = KMeans(n_clusters = number_of_clusters)

    km.fit(tf_idf_matrix)

    # print(km.fit)

    # lets print clustering info ...

    results = pd.DataFrame()
    results['document_text'] = corpus
    results['cluster_id'] = km.labels_

    print("\n Results of clustering:\n", results)

    # Evaluate Clustering Results: Pre, Recall, F1, Support

    actual_cluster = [0,1,1,0,1,0,0,0]
    pre_clusters =  km.labels_

    pre,rec,fsco,supp = score(actual_cluster, pre_clusters)

    print("\n Clustering Evaluation Results:\n")
    print("pre: {}".format(pre))
    print("rec: {}".format(rec))
    print("fsco: {}".format(fsco))
    print("supp: {}".format(supp))

main()