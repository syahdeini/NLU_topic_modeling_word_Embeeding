# coding: utf-8

import gensim
import math
from copy import copy
import pdb
'''
(f) helper class, do not modify.
provides an iterator over sentences in the provided BNC corpus
input: corpus path to the BNC corpus
input: n, number of sentences to retrieve (optional, standard -1: all)
'''
class BncSentences:
    def __init__(self, corpus, n=-1):
        self.corpus = corpus
        self.n = n
    
    def __iter__(self):
        n = self.n
        ret = []
        for line in open(self.corpus):
            line = line.strip().lower()
            if line.startswith("<s "):
                ret = []
            elif line.strip() == "</s>":
                if n > 0:
                    n -= 1
                if n == 0:
                    break
                yield copy(ret)
            else:
                parts = line.split("\t")
                if len(parts) == 3:
                    word = parts[-1]
                    idx = word.rfind("-")
                    word, pos = word[:idx], word[idx+1:]
                    if word in ['thus', 'late', 'often', 'only', 'usually', 'however', 'lately', 'absolutely', 'hardly', 'fairly', 'near', 'similarly', 'sooner', 'there', 'seriously', 'consequently', 'recently', 'across', 'softly', 'together', 'obviously', 'slightly', 'instantly', 'well', 'therefore', 'solely', 'intimately', 'correctly', 'roughly', 'truly', 'briefly', 'clearly', 'effectively', 'sometimes', 'everywhere', 'somewhat', 'behind', 'heavily', 'indeed', 'sufficiently', 'abruptly', 'narrowly', 'frequently', 'lightly', 'likewise', 'utterly', 'now', 'previously', 'barely', 'seemingly', 'along', 'equally', 'so', 'below', 'apart', 'rather', 'already', 'underneath', 'currently', 'here', 'quite', 'regularly', 'elsewhere', 'today', 'still', 'continuously', 'yet', 'virtually', 'of', 'exclusively', 'right', 'forward', 'properly', 'instead', 'this', 'immediately', 'nowadays', 'around', 'perfectly', 'reasonably', 'much', 'nevertheless', 'intently', 'forth', 'significantly', 'merely', 'repeatedly', 'soon', 'closely', 'shortly', 'accordingly', 'badly', 'formerly', 'alternatively', 'hard', 'hence', 'nearly', 'honestly', 'wholly', 'commonly', 'completely', 'perhaps', 'carefully', 'possibly', 'quietly', 'out', 'really', 'close', 'strongly', 'fiercely', 'strictly', 'jointly', 'earlier', 'round', 'as', 'definitely', 'purely', 'little', 'initially', 'ahead', 'occasionally', 'totally', 'severely', 'maybe', 'evidently', 'before', 'later', 'apparently', 'actually', 'onwards', 'almost', 'tightly', 'practically', 'extremely', 'just', 'accurately', 'entirely', 'faintly', 'away', 'since', 'genuinely', 'neatly', 'directly', 'potentially', 'presently', 'approximately', 'very', 'forwards', 'aside', 'that', 'hitherto', 'beforehand', 'fully', 'firmly', 'generally', 'altogether', 'gently', 'about', 'exceptionally', 'exactly', 'straight', 'on', 'off', 'ever', 'also', 'sharply', 'violently', 'undoubtedly', 'more', 'over', 'quickly', 'plainly', 'necessarily']:
                        pos = "r"
                    if pos == "j":
                        pos = "a"
                    ret.append(gensim.utils.any2unicode(word + "." + pos))

'''
(a) function load_corpus to read a corpus from disk
input: vocabFile containing vocabulary
input: contextFile containing word contexts
output: id2word mapping word IDs to words
output: word2id mapping words to word IDs
output: vectors for the corpus, as a list of sparse vectors
'''
def load_corpus(vocabFile, contextFile):
    id2word = {}
    word2id = {}
    idx=0
    vectors = [] 
    for line in open(vocabFile):
        line = line.strip()
        word = line #[:-2]+".n"
        id2word[idx]=word
        word2id[word]=idx
        idx+=1
    idx=0
    full_vec = []
    for line in open(contextFile):
        element = line.strip().split()
        oc_count = int(element[0])
        a_vec = []
        for _e in element[1:]:
            nh,count = _e.split(":")
            a_vec.append((nh,int(count)))
        full_vec.append(a_vec)
    return id2word, word2id, full_vec

'''
(b) function cosine_similarity to calculate similarity between 2 vectors
input: vector1
input: vector2
output: cosine similarity between vector1 and vector2 as a real number
'''


def convert_to_sparse_vec(vector1,vector2):
        svec1=[]
        svec2=[]
        for vec1 in enumerate(vector1):
            svec1.append((vec1[0],vec1[1]))
        for vec2 in enumerate(vector2):
            svec2.append((vec2[0],vec2[1]))
        return svec1,svec2




    
def cosine_similarity(vector1, vector2):
  
    dict_vec1={}
    dict_vec2={}
    len_vec1=0
    len_vec2=0
    if type(vector1[0]) is not tuple:
        vector1, vector2 = convert_to_sparse_vec(vector1, vector2)
    
    for vec in vector1:
            dict_vec1[vec[0]]=vec[1]
            len_vec1=len_vec1+(vec[1]*vec[1])
    for vec in vector2:
            dict_vec2[vec[0]]=vec[1]
            len_vec2=len_vec2+(vec[1]*vec[1])
    sums=0
    for vec in dict_vec1:
        if vec in dict_vec2:
            sums = sums + dict_vec1[vec]*dict_vec2[vec]
    return sums/(math.sqrt(len_vec1)*math.sqrt(len_vec2))


'''
(d) function tf_idf to turn existing frequency-based vector model into tf-idf-based vector model
input: freqVectors, a list of frequency-based vectors
output: tfIdfVectors, a list of tf-idf-based vectors
'''

def get_dfi(freqVectors):
    dfi_dict={}
    for doc in freqVectors:
        for key_v in doc:
                if key_v[0] in dfi_dict:
                    dfi_dict[key_v[0]]+=1
                else:
                    dfi_dict[key_v[0]]=1
    return dfi_dict

def tf_idf(freqVectors):
    tfIdfVectors = []
    num_doc=len(freqVectors)
    dfi=get_dfi(freqVectors)
    # sparse vector
    for doc in freqVectors:
        temp_list=[]
        for voc in doc:
            tfidf=(1+math.log(int(voc[1]),2))*(1+math.log(float(num_doc)/dfi[voc[0]],2))
            temp_list.append((voc[0],tfidf))
        tfIdfVectors.append(temp_list)
    # your code here    
    return tfIdfVectors

'''
(f) function word2vec to build a word2vec vector model with 100 dimensions and window size 5
'''
def word2vec(corpus, learningRate, downsampleRate, negSampling):
    # your code here
    train_sentence=BncSentences(corpus=corpus,n=50000)
    model = gensim.models.Word2Vec(train_sentence,size=100,window=5,alpha=learningRate,sample=downsampleRate,negative=negSampling,workers=3)
    return model

'''
(h) function lda to build an LDA model with 100 topics from a frequency vector space
input: vectors
input: wordMapping mapping from word IDs to words
output: an LDA topic model with 100 topics, using the frequency vectors
'''
def lda(vectors, wordMapping):
    model = gensim.models.ldamodel.LdaModel(corpus=vectors, id2word=wordMapping,num_topics=100,update_every=0,passes=10)
    return model

'''
(j) function get_topic_words, to get words in a given LDA topic
input: ldaModel, pre-trained Gensim LDA model
input: topicID, ID of the topic for which to get topic words
input: wordMapping, mapping from words to IDs (optional)
'''
def get_topic_words(ldaModel, topicID):
    # your code here
    topic_given_word=ldaModel.show_topic(topicID,topn=30)
    return topic_given_word



if __name__ == '__main__':
    import sys
    ## for test case only ##
    _a1="/afs/inf.ed.ac.uk/group/project/nlu/assignment1/data/vocabulary.txt"
    _a2="/afs/inf.ed.ac.uk/group/project/nlu/assignment1/data/word_contexts.txt"
    id2word, word2id, vectors = load_corpus(_a1, _a2)
        
    part = sys.argv[1].lower()
    
    # these are indices for house, home and time in the data. Don't change.
    house_noun = 80
    home_noun = 143
    time_noun = 12
    
    # this can give you an indication whether part a (loading a corpus) works.
    # not guaranteed that everything works. 
    if part == "a":
        print("(a): load corpus")
        try:
            id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
            if not id2word:
                print("\tError: id2word is None or empty")
                exit()
            if not word2id:
                print("\tError: id2word is None or empty")
                exit()
            if not vectors:
                print("\tError: id2word is None or empty")
                exit()
            print("\tPass: load corpus from file")
        except Exception as e:
            print("\tError: could not load corpus from disk")
            print(e)
        
        try:
            if not id2word[house_noun] == "house.n" or not id2word[home_noun] == "home.n" or not id2word[time_noun] == "time.n":
                print("\tError: id2word fails to retrive correct words for ids")
            else:
                print("\tPass: id2word")
        except Exception:
            print("\tError: Exception in id2word")
            print(e)
        
        try:
            if not word2id["house.n"] == house_noun or not word2id["home.n"] == home_noun or not word2id["time.n"] == time_noun:
                print("\tError: word2id fails to retrive correct ids for words")
            else:
                print("\tPass: word2id")
        except Exception:
            print("\tError: Exception in word2id")
            print(e)
    
    # this can give you an indication whether part b (cosine similarity) works.
    # these are very simple dummy vectors, no guarantee it works for our actual vectors.
    if part == "b":
        import numpy
        print("(b): cosine similarity")
        try:
            cos = cosine_similarity([(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)])
            print(cos)
            if not numpy.isclose(0.5, cos):
                print("\tError: sparse expected similarity is 0.5, was {0}".format(cos))
            else:
                print("\tPass: sparse vector similarity")
        except Exception:
            print("\tError: failed for sparse vector")
        try:
            cos = cosine_similarity([1, 0, 1, 0, 2], [1, 2, 0, 0, 1])
            if not numpy.isclose(0.5, cos):
                print("\tError: full expected similarity is 0.5, was {0}".format(cos))
            else:
                print("\tPass: full vector similarity")
        except Exception:
            print("\tError: failed for full vector")

    # you may complete this part to get answers for part c (similarity in frequency space)
    if part == "c":
        print("(c) similarity of house, home and time in frequency space")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        house_vec=vectors[word2id["house.n"]]
        home_vec=vectors[word2id["home.n"]]
        time_vec=vectors[word2id["time.n"]]
        print("house and home",cosine_similarity(house_vec,home_vec))
        print("house and time",cosine_similarity(house_vec,time_vec))
        print("home and time",cosine_similarity(time_vec,home_vec))
        # your code here
         
    # this gives you an indication whether your conversion into tf-idf space works.
    # this does not test for vector values in tf-idf space, hence can't tell you whether tf-idf has been implemented correctly
    if part == "d":
        print("(d) converting to tf-idf space")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        try:
            tfIdfSpace = tf_idf(vectors)
            if not len(vectors) == len(tfIdfSpace):
                print("\tError: tf-idf space does not correspond to original vector space")
            else:
                print("\tPass: converted to tf-idf space")
        except Exception as e:
            print("\tError: could not convert to tf-idf space")
            print(e)
    
    # you may complete this part to get answers for part e (similarity in tf-idf space)
    if part == "e":
        print("(e) similarity of house, home and time in tf-idf space")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        tfIdfSpace = tf_idf(vectors)
        house_vec=tfIdfSpace[word2id["house.n"]]
        home_vec=tfIdfSpace[word2id["home.n"]]
        time_vec=tfIdfSpace[word2id["time.n"]]
        print("house and home",cosine_similarity(house_vec,home_vec))
        print("house and time",cosine_similarity(house_vec,time_vec))
        print("hom  e and time",cosine_similarity(time_vec,home_vec))
        # your code here
        # your code here
    
    # you may complete this part for the first part of f (estimating best learning rate, sample rate and negative samplings)
    if part == "f1":
        print("(f1) word2vec, estimating best learning rate, sample rate, negative sampling")
        # for lrate in [0.01,0.035,0.05]:
        #     for down_rate in [0.01,0.003,0.0005,0.00001]:
        #         for neg_samp in [1,5,7,10]:    
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        for lrate in [0.01,0.03,0.05]:
            for down_rate in [0.1,0.005,0.00001]:
                for neg_samp in [1,5,10]:   
                    model=word2vec("/afs/inf.ed.ac.uk/group/project/nlu/assignment1/data/bnc.vert", learningRate=lrate,downsampleRate=down_rate,negSampling=neg_samp)
                    acc=model.accuracy("/afs/inf.ed.ac.uk/group/project/nlu/assignment1/data/accuracy_test.txt")
                    print("#### for lrate = ",lrate,", with down_rate = ",down_rate,", with negative sample = ",neg_samp)
                    print(acc)
                    print("#########")
        # your code here
    
    # you may complete this part for the second part of f (training and saving the actual word2vec model)
    if part == "f2":
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        print("(f2) word2vec, building full model with best parameters. May take a while.")
        lrate=0.01
        down_rate=0.05
        neg_samp=5
        model=word2vec("/afs/inf.ed.ac.uk/group/project/nlu/assignment1/data/bnc.vert", learningRate=lrate,downsampleRate=math.pow(0.1,down_rate),negSampling=neg_samp)
        model.save("word2vec_m")
               #
 
        # your code here
    
    # you may complete this part to get answers for part g (similarity in your word2vec model)
    if part == "g":
        print("(g): word2vec based similarity")
        model=gensim.models.Word2Vec.load("word2vec_m")
        house_vec=model["house.n"]
        home_vec=model["home.n"]
        time_vec=model["time.n"]
        print("house and home",cosine_similarity(house_vec,home_vec))
        print("house and time",cosine_similarity(house_vec,time_vec))
        print("home and time",cosine_similarity(time_vec,home_vec))

        # your code here
    
    # you may complete this for part h (training and saving the LDA model)
    if part == "h":
        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        print("(h) LDA model")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        model=lda(vectors,id2word)
        # your code here
        model.save("lda_m")
                
    
    # you may complete this part to get answers for part i (similarity in your LDA model)
    if part == "i":
        print("(i): lda-based similarity")
        model = gensim.models.LdaModel.load("lda_m")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        house_vec=model[vectors[word2id["house.n"]]]
        home_vec=model[vectors[word2id["home.n"]]]
        time_vec=model[vectors[word2id["time.n"]]]
        print("house and home",cosine_similarity(house_vec,home_vec))
        print("house and time",cosine_similarity(house_vec,time_vec))
        print("home and time",cosine_similarity(time_vec,home_vec))
 
        # your code here

    # you may complete this part to get answers for part j (topic words in your LDA model)
    if part == "j":
        print("(j) get topics from LDA model")
        id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
        model = gensim.models.LdaModel.load("lda_m")
        for i in range(100):
            print('----------')
            print('topic number-',i,' = ',get_topic_words(model,i))


        # your code here
