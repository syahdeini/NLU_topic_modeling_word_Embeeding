# coding: utf-8

from question1 import *
import json
import pdb
'''
helper class to load a thesaurus from disk
input: thesaurusFile, file on disk containing a thesaurus of substitution words for targets
output: the thesaurus, as a mapping from target words to lists of substitution words
'''
def load_thesaurus(thesaurusFile):
    thesaurus = {}
    with open(thesaurusFile) as inFile:
        for line in inFile.readlines():
            word, subs = line.strip().split("\t")
            thesaurus[word] = subs.split(" ")
    return thesaurus

'''
(a) function addition for adding 2 vectors
input: vector1
input: vector2
output: addVector, the resulting vector when adding vector1 and vector2
'''
def conv_dict_sparse(_dict):
        sparse=[]
        for key in _dict:
            sparse.append((key,_dict[key]))
        return sparse

        
def addition(vector1, vector2):
    # your code here
    sum_vec = {}
    if type(vector1[0]) is tuple:
            for vec1 in vector1:
                    sum_vec[vec1[0]]=vec1[1]
            for vec2 in vector2:
                    if vec2[0] in sum_vec:
                        sum_vec[vec2[0]] += vec2[1]
                    else:
                        sum_vec[vec2[0]] = vec2[1]
    else: #full vector
        idx=0
        for vec1,vec2 in zip(vector1,vector2):
            sum_vec[idx]=vec1+vec2
            idx+=1
    # sparse vector:
    return conv_dict_sparse(sum_vec)

'''
(a) function multiplication for multiplying 2 vectors
input: vector1
input: vector2
output: mulVector, the resulting vector when multiplying vector1 and vector2
'''
def multiplication(vector1, vector2):
    # your code here
    mul_vec1={}
    mul_vec2={}
    mul_vec={}
    if type(vector1[0]) is tuple:
          for vec1 in vector1:
              mul_vec1[vec1[0]]=vec1[1]
          for vec2 in vector2:
              mul_vec2[vec2[0]]=vec2[1]

          for vec in mul_vec1:
            if vec in mul_vec2:
                    mul_vec[vec]=mul_vec1[vec]*mul_vec2[vec]
    else: #full vector
        idx=0
        for vec1,vec2 in zip(vector1,vector2):
            mul_vec[idx]=vec1*vec2
            idx+=1
    return conv_dict_sparse(mul_vec)

'''
(d) function prob_z_given_w to get probability of LDA topic z, given target word w
input: ldaModel
input: topicID as an integer
input: wordVector in frequency space
output: probability of the topic with topicID in the ldaModel, given the wordVector
'''
def prob_z_given_w(ldaModel, topicID, wordVector):
    topic_given_word = ldaModel[[(int(x),y) for x,y in wordVector]]
    for topic,topic_p in topic_given_word:
            if topic == topicID:
                    return topic_p
    return 0

'''
(d) function prob_w_given_z to get probability of target word w, given LDA topic z
input: ldaModel
input: targetWord as a string
input: topicID as an integer
output: probability of the targetWord, given the topic with topicID in the ldaModel
'''
def prob_w_given_z(ldaModel, targetWord, topicID):
    # your code herea
    word_given_topic = ldaModel.show_topic(topicID,topn=ldaModel.num_terms)
    for word,word_p in word_given_topic:
            if str(word) == str(targetWord):
                return word_p
    return 0

'''
(f) get the best substitution word in a given sentence, according to a given model (tf-idf, word2vec, LDA) and type (addition, multiplication, lda)
input: jsonSentence, a string in json format
input: thesaurus, mapping from target words to candidate substitution words
input: word2id, mapping from vocabulary words to word IDs
input: model, a vector space, Word2Vec or LDA model
input: frequency vectors, original frequency vectors (for querying LDA model)
input: csType, a string indicating the method of calculating context sensitive vectors: "addition", "multiplication", or "lda"
output: the best substitution word for the jsonSentence in the given model, using the given csType
'''

def load_json(_file):
    f=open(_file,"r")
    _list = list()
    for l in f:
        _list.append(json.loads(l))
    return _list

#def load_thesaurus(_file):
 #   _file = open(_file,'r')
  #  thes_dict = dict()
   # for l in _file:
    #    _tl = l.split()    
     #   thes_dict[l[0]]= _tl[1:]
   # return thes_dict
    
def get_context(pos,sentence):
    con_words=[]
    pos = int(pos)
    sentence=sentence['sentence'].split()
    for i in range(5):
        if pos-i > 0: 
            con_words.append(sentence[pos-i])
        if pos+i < len(sentence):
            con_words.append(sentence[pos+i])
    return con_words

def get_vector(word, model, word2id, freqVector):
    if type(model) is gensim.models.Word2Vec:
        return model[word],"word2vec" # return vector word and filename
    elif type(model) is gensim.models.ldamodel.LdaModel:
        return model[freqVector[word2id[word]]],"lda"
    else: #must be vector space
        return model[word2id[word]],"tf-idf"

def lda_func(ldaModel,target_word,wordVec,context_word):
     z_t = []
     c_z = []
     den = 0
     for topicID in range(100):
        p_z_w = prob_z_given_w(ldaModel,topicID,wordVec)
        p_w_z = prob_w_given_z(ldaModel,context_word,topicID)
        z_t.append(p_z_w)
        c_z.append(p_w_z)
        den += p_z_w*p_z_w
     v_t_c=[]
     for topicID in range(100):
        val = (z_t[topicID]*c_z[topicID])/den
        if val!=0:  
            v_t_c.append((topicID,val))
     return v_t_c
    

def do_operation(v_t,v_c,csType,model=None,t_word="",freqVec=[],c_word=""):
    # calculate cosine similarity (v(w),v(t,c))
    if csType == "addition":
        return addition(v_t,v_c)
    # (c) use multiplication to get context sensitive vectors
    elif csType == "multiplication":
        return multiplication(v_t,v_c)
    # (d) use LDA to get context sensitive vectors
    elif csType == "lda":
        return lda_func(model,t_word,freqVec,c_word)


def best_substitute(jsonSentence, thesaurus, word2id, model, frequencyVectors, csType):
    # V(c) is generated by model    
    # (b) use addition to get context sensitive vectors
    
    file_d = None
    sentence=jsonSentence
    target_word = sentence['target_word']
    sentence_id = sentence['id']
    # leave the prediction empty if our target_wrod faile to predict.
    max_subs_word = ""
    if target_word in word2id:
       wordVec = frequencyVectors[word2id[target_word]]
       pos_target_word = int(sentence['target_position'])
       candidate_subs = thesaurus[target_word]
       contexts_word = get_context(pos_target_word,sentence)
       v_t,_ = get_vector(target_word, model, word2id, frequencyVectors)
       # for each subsitute word we try to sum over all the context
       # then save all the result and find the maximum one
       max_subs_score = -100
       max_subs_word = "cyka"
       for subs in candidate_subs:    
            v_w, filename = get_vector(subs, model, word2id, frequencyVectors)
            score_each_subs = 0    
            for c_word in contexts_word:            
                # ignore if if the context word is not present in the vocabularys
                if c_word not in word2id:
                    continue
                v_c,_ = get_vector(c_word, model, word2id, frequencyVectors)
                v_t_c = do_operation(v_t,v_c,csType,model,target_word,wordVec,c_word)
                if v_t_c == 0:
                    score=0
                else:
                    score=cosine_similarity(v_w,v_t_c)
                score_each_subs += score
            # then we need to find the maximum score for each subsitute word
            if score_each_subs > max_subs_score:
               max_subs_word = subs

       # writing the best subsitutiton into a file (appending)
       if not file_d:
           filename = filename + "_" + csType + ".txt"
       file_d = open(filename, "a")        
       stuff_to_write = target_word + " " + sentence_id + " :: " + max_subs_word+"\n"
       file_d.write(stuff_to_write)
                


if __name__ == "__main__":
    import sys
    
    part = sys.argv[1]
    
    # this can give you an indication whether part a (vector addition and multiplication) works.
    if part == "a":
        print("(a): vector addition and multiplication")
        v1, v2, v3 , v4 = [(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)], [1, 0, 1, 0, 2], [1, 2, 0, 0, 1]
        try:
            if not set(addition(v1, v2)) == set([(0, 2), (2, 1), (4, 3), (1, 2)]):
                print("\tError: sparse addition returned wrong result")
            else:
                print("\tPass: sparse addition")
        except Exception as e:
            print("\tError: exception raised in sparse addition")
            print(e)
        try:
            if not set(multiplication(v1, v2)) == set([(0,1), (4,2)]):
                print("\tError: sparse multiplication returned wrong result")
            else:
                print("\tPass: sparse multiplication")
        except Exception as e:
            print("\tError: exception raised in sparse multiplication")
            print(e)
        try:
            addition(v3,v4)
            print("\tPass: full addition")
        except Exception as e:
            print("\tError: exception raised in full addition")
            print(e)
        try:
            multiplication(v3,v4)
            print("\tPass: full multiplication")
        except Exception as e:
            print("\tError: exception raised in full addition")
            print(e)
    
    # you may complete this to get answers for part b (best substitution words with tf-idf and word2vec, using addition)
    if part == "b":
        print("(b) using addition to calculate best substitution words")
        # your code here
        sentence_file ="/afs/inf.ed.ac.uk/group/project/nlu/assignment1/data/test.txt"
        thesaurus_file="/afs/inf.ed.ac.uk/group/project/nlu/assignment1/data/test_thesaurus.txt"
        sentences=load_json(sentence_file)
        thesaurus=load_thesaurus(thesaurus_file)
        id2word,word2id,vectors=load_corpus(sys.argv[2], sys.argv[3])
        model = tf_idf(vectors)
        for sent in sentences:
                best_substitute(sent,thesaurus,word2id,model,vectors,"addition") 
        # you may complete this to get answers for part c (best substitution words with tf-idf and word2vec, using multiplication)
    if part == "c":
        print("(c) using multiplication to calculate best substitution words")
        sentence_file ="/afs/inf.ed.ac.uk/group/project/nlu/assignment1/data/test.txt"
        thesaurus_file="/afs/inf.ed.ac.uk/group/project/nlu/assignment1/data/test_thesaurus.txt"
        sentences=load_json(sentence_file)
        thesaurus=load_thesaurus(thesaurus_file)
        id2word,word2id,vectors=load_corpus(sys.argv[2], sys.argv[3])
        model = tf_idf(vectors)
        for sent in sentences:
                best_substitute(sent,thesaurus,word2id,model,vectors,"multiplication") 
    # this can give you an indication whether your part d1 (P(Z|w) and P(w|Z)) works
    if part == "d":
        print("(d): calculating P(Z|w) and P(w|Z)")
        print("\tloading corpus")
        id2word,word2id,vectors=load_corpus(sys.argv[2], sys.argv[3])
        print("\tloading LDA model")
        ldaModel = gensim.models.ldamodel.LdaModel.load("lda_m")
        houseTopic = ldaModel[vectors[word2id["house.n"]]][0][0]
        try:
            if prob_z_given_w(ldaModel, houseTopic, vectors[word2id["house.n"]]) > 0.0:
                print("\tPass: P(Z|w)")
            else:
                print("\tFail: P(Z|w)")
        except Exception as e:
            print("\tError: exception during P(Z|w)")
            print(e)
        try:
            if prob_w_given_z(ldaModel, "house.n", houseTopic) > 0.0:
                print("\tPass: P(w|Z)")
            else:
                print("\tFail: P(w|Z)")
        except Exception as e:
            print("\tError: exception during P(w|Z)")
            print(e)
    
    # you may complete this to get answers for part d2 (best substitution words with LDA)
    if part == "e":
        print("(e): using LDA to calculate best substitution words")
        # your code here
        sentence_file ="/afs/inf.ed.ac.uk/group/project/nlu/assignment1/data/test.txt"
        thesaurus_file="/afs/inf.ed.ac.uk/group/project/nlu/assignment1/data/test_thesaurus.txt"
        sentences=load_json(sentence_file)
        thesaurus=load_thesaurus(thesaurus_file)
        model = gensim.models.ldamodel.LdaModel.load("lda_m")
        id2word,word2id,vectors=load_corpus(sys.argv[2], sys.argv[3])
        for sent in sentences:
                best_substitute(sent,thesaurus,word2id,model,vectors,"lda") 

