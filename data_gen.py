import os
import pandas as pd
import numpy as np
import sys

import time
import re 

import tagme
from typing import List, Set

tagme.GCUBE_TOKEN = '8ea45e82-c07f-452a-867a-e16fed5be4bb-843339462'

def getIntersection(abstract_entity : List[str], title_entity : List[str]) -> Set[str]:

    return set(abstract_entity).intersection(title_entity)


def getDiff(abstract_entity : List[str], title_entity : List[str]) -> Set[str]:

    return set(abstract_entity).difference(title_entity)

count = 0
total = 0

def getEntities(sentence : str, threshold : float = 0.1) -> List[str]:
    #result = []

    global count, total

    print(total)
    total += 1

    if sentence == '':
        return []

    try:
        annotations = tagme.annotate(sentence)
        result = list(annotations.get_annotations(threshold))
        return list(map(lambda x : x.entity_title, result))
    except BaseException as e:
        print('Exception', e)
        print('Corresponding Sentence:')
        print(count, total)
        count += 1
        return []


#data_path = '../scholar/sample-S2-records'
#data_path = 'sample-S2-records'
data_path = 'd10000-aa'
vector_path = 'words_scholar'
data = None
entities = None
words = None
entity_description = None

def preprocess(just_part = False) : 

    global data, entities

    data = pd.read_json(data_path, lines = True)
    #data = data[:2]

    data = data[['paperAbstract', 'title', 'id']]

    if just_part:
        return

    start_time = time.clock()

    data['abstract_entity'] = data['paperAbstract'].map(lambda x : getEntities(x))
    data['title_entity'] = data['title'].map(lambda x : getEntities(x))

    end_time = time.clock()
    print('Entity Linking costs ', end_time - start_time, 's.')
    print('Finish Entity Linking...')

    start_time = time.clock()
    data['pos_entity'] = data[['abstract_entity', 'title_entity']].apply(lambda x : getIntersection(x.abstract_entity, x.title_entity), axis = 1)
    data['neg_entity'] = data[['abstract_entity', 'title_entity']].apply(lambda x : getDiff(x.abstract_entity, x.title_entity), axis = 1)
    data['pos_entity'] = data['pos_entity'].map(lambda x : list(x))
    data['neg_entity'] = data['neg_entity'].map(lambda x : list(x))

    end_time = time.clock()
    print('Salience Entity costs ', end_time - start_time, 's.')

    data['paperAbstract'] = data['paperAbstract'].map(lambda x : x.replace(',', '').lower())
    data['paperAbstract'] = data['paperAbstract'].map(lambda x : x.replace('.', ''))

    data['title'] = data['title'].map(lambda x : x.replace(',', '').lower())
    data['title'] = data['title'].map(lambda x : x.replace('.', ''))
    data['title'] = data['title'].map(lambda x : x.replace('[', ''))
    data['title'] = data['title'].map(lambda x : x.replace(']', ''))
    data['title'] = data['title'].map(lambda x : x.replace(':', ''))

    total_entities = pd.concat([data['abstract_entity'], data['title_entity']], ignore_index = True)

    entity_set = set()
    for index, value in total_entities.iteritems(): 
        entity_set.update(value)

    entities = pd.Series(list(entity_set))
    entities = entities.map(lambda x : x.replace(' ', '_').lower())

    return data, entities

def saveForWord2Vec(path : str):

    global data, entity_description

    with open(path, 'w') as file:
        for index, value in entity_description.iterrows():
            file.write(index)
            file.write('\n')
            for i in value:
                file.write(i)
                file.write(' ')
            file.write('\n')
        for i in data['paperAbstract']:
            file.write(i)
            file.write('\n')
        for i in data['title']:
            file.write(i)
            file.write('\n')


def saveEntity():

    global entities, entity_description

    if isinstance(entities, pd.Series):
        entities.to_csv('entity.csv', index = False)
    else:
        print('Entities Does not Exist.')

    if isinstance(entity_description, pd.DataFrame):
        entity_description.to_csv('entity_description.csv')
    else:
        print('Entities Does not Exist.')


def saveData():

    global data

    if isinstance(data, pd.DataFrame):
        data.to_csv('data.csv', index = False)
    else:
        print('Data Does not Exist.')


def saveWords():

    global words

    if isinstance(words, pd.DataFrame):
        words.to_csv('words.csv')
    else:
        print('Words Does not Exist.') 


def save():

    saveEntity()
    saveData()
    saveWords()


def readVectors(path : str) -> pd.DataFrame:

    global words

    with open(path, 'r') as file:
        
        lines = file.readlines()
        lines = list(map(lambda x : x.rstrip(), lines))

        first_line = lines[0].split(' ')
        vocab_size = int(first_line[0])
        vector_size = int(first_line[1])

        word_vector = []
        lines = lines[1:]
        for l in lines:
            t = l.split(' ')
            t[0] = t[0].replace(' ', '_').lower()
            for i in range(vector_size):
                t[i + 1] = float(t[i + 1])
            word_vector.append(t)
        words =  pd.DataFrame(word_vector)
        words = words.set_index(0)
        
    return words

L = 10


def prepare(art_path, entity_path, entity_reform_path, description_path, w2v_vec_path):
    global data
    if art_path.split('.')[-1] == 'csv':
        data = pd.read_csv(art_path, keep_default_na=False)
    else:
        data = pd.read_json(art_path, lines=True)

    data = data[['paperAbstract', 'title', 'id']]

    entity_data = pd.read_csv(entity_path, keep_default_na=False)
    result = []
    for index, erow in entity_data.iterrows():
        result.append([erow['id'], erow['title'].strip().split(','), erow['paperAbstract'].strip().split(',')])
    result = pd.DataFrame(result)
    result = result.set_index(0)
    result.to_json(entity_reform_path)

    _ = readSupplyData(entity_reform_path)
    _ = addDescription(description_path)
    _ = readVectors(w2v_vec_path)

def getDocumentData():

    global data, words, L, entity_description

    for _, value in data.iterrows():

        if '' in value.pos_entity:
            value.pos_entity.remove('')
        if '' in value.neg_entity:
            value.neg_entity.remove('')

        pos_e_shape = (len(value.pos_entity), 128)
        neg_e_shape = (len(value.neg_entity), 128)

        if value.pos_entity == ['']:
            pos_e_shape = (0, 128)
        if value.neg_entity == ['']:
            neg_e_shape = (0, 128)

        pos_ed_shape = (pos_e_shape[0], L, 128)
        neg_ed_shape = (neg_e_shape[0], L, 128)

        #value.pos_entity = list(map(lambda x : x if x is not '' else '</s>', value.pos_entity))
        #value.neg_entity = list(map(lambda x : x if x is not '' else '</s>', value.neg_entity))

        pos_e = np.ones(pos_e_shape, dtype = np.float)
        for i in range(pos_e_shape[0]):
            pos_e[i] = words.loc[value.pos_entity[i]]

        pos_ed = np.ones(pos_ed_shape, dtype = np.float)
        for i in range(pos_ed_shape[0]):
            desciption = entity_description.loc[value.pos_entity[i]]
            desciption = list(map(lambda x : x if x is not '' else '</s>', desciption))
            for j in range(pos_ed_shape[1]):
                pos_ed[i][j] = words.loc[desciption[j].lower()]

        neg_e = np.ones(neg_e_shape, dtype = np.float)
        for i in range(neg_e_shape[0]):
            neg_e[i] = words.loc[value.neg_entity[i]]

        neg_ed = np.ones(neg_ed_shape, dtype = np.float)
        for i in range(neg_ed_shape[0]):
            desciption = entity_description.loc[value.neg_entity[i]]
            desciption = list(map(lambda x : x if x is not '' else '</s>', desciption))
            for j in range(neg_ed_shape[1]):
                neg_ed[i][j] = words.loc[desciption[j].lower()]
        
        abstract_words = value.paperAbstract.replace(',', '').lower()
        abstract_words = abstract_words.replace('.', '')
        abstract_words = abstract_words.strip()
        abstract_words = abstract_words.replace('\n', ' ')#re.split(' |\n', abstract_words)
        abstract_words = abstract_words.split(' ')
        abstract_words = list(map(lambda x : x if x is not '' else '</s>', abstract_words))
        doc_w_shape = (len(abstract_words), 128)
        doc_w = np.ones(doc_w_shape, dtype = np.float)
        for i in range(doc_w_shape[0]):
            try:
                doc_w[i] = words.loc[abstract_words[i]]
            except KeyError:
                doc_w[i] = words.loc['</s>']
        
        doc_e_shape = (neg_e_shape[0] + pos_e_shape[0], 128)
        doc_e = np.concatenate([pos_e, neg_e])

        doc_ed_shape = (neg_e_shape[0] + pos_e_shape[0], L, 128)
        doc_ed = np.concatenate([pos_ed, neg_ed])
        

        yield pos_e.astype(np.float32), \
              neg_e.astype(np.float32), \
              pos_ed.astype(np.float32), \
              neg_ed.astype(np.float32), \
              doc_w.astype(np.float32), \
              doc_e.astype(np.float32), \
              doc_ed.astype(np.float32)


def addDescription(path : str):

    global entity_description

    with open(path, 'r') as file:

        result = []
        lines = file.readlines()

        for l in lines:
            l = l.rstrip().split('`')
            tmp = [l[0].replace(' ', '_').lower()]
            tmp.extend(l[1].split(' '))
            result.append(tmp)
        entity_description = pd.DataFrame(result)
        entity_description= entity_description.set_index(0)
        entity_description = entity_description[entity_description.columns[:10]]
        entity_description = entity_description.applymap(lambda x : x if x is not '' and x is not None else '</s>')
        entity_description = entity_description.applymap(lambda x : x.rstrip(',').lower())
        entity_description = entity_description[~entity_description.index.duplicated(keep='first')]
    return entity_description


def reformatFile(path : str):

    with open(path, encoding='utf-8', mode = 'r') as file:
        lines = file.readlines()
        result = []
        for line in lines:
            line = line.strip().split('`')
            result.append([line[0], line[1].split(','), line[2].split(',')])
        #print(result)
        data = pd.DataFrame(result)
        data = data.set_index(0)
        #print(data)
        data.to_json(path + '.json')



def readSupplyData(path: str):

    global data, entities

    t = pd.read_json(path)

    data = data[data.id.isin(t.index)]

    start_time = time.clock()
    data['abstract_entity'] =  data['id'].map(lambda x : t.loc[x][2])
    data['title_entity'] = data['id'].map(lambda x : t.loc[x][1])

    end_time = time.clock()
    print('Entity Linking costs ', end_time - start_time, 's.')
    print('Finish Entity Linking...')

    start_time = time.clock()
    data['pos_entity'] = data[['abstract_entity', 'title_entity']].apply(lambda x : getIntersection(x.abstract_entity, x.title_entity), axis = 1)
    data['neg_entity'] = data[['abstract_entity', 'title_entity']].apply(lambda x : getDiff(x.abstract_entity, x.title_entity), axis = 1)
    data['pos_entity'] = data['pos_entity'].map(lambda x : list(x))
    data['pos_entity'] = data['pos_entity'].map(lambda x : list(map(lambda x: x.replace(' ', '_').lower(), x)))
    data['neg_entity'] = data['neg_entity'].map(lambda x : list(x))
    data['neg_entity'] = data['neg_entity'].map(lambda x : list(map(lambda x: x.replace(' ', '_').lower(), x)))

    end_time = time.clock()
    print('Salience Entity costs ', end_time - start_time, 's.')

    data['paperAbstract'] = data['paperAbstract'].map(lambda x : x.replace(',', '').lower())
    data['paperAbstract'] = data['paperAbstract'].map(lambda x : x.replace('.', ''))

    data['title'] = data['title'].map(lambda x : x.replace(',', '').lower())
    data['title'] = data['title'].map(lambda x : x.replace('.', ''))
    data['title'] = data['title'].map(lambda x : x.replace('[', ''))
    data['title'] = data['title'].map(lambda x : x.replace(']', ''))
    data['title'] = data['title'].map(lambda x : x.replace(':', ''))

    total_entities = pd.concat([data['abstract_entity'], data['title_entity']], ignore_index = True)

    entity_set = set()
    for index, value in total_entities.iteritems(): 

        entity_set.update(value)

    entity_set = list(map(lambda x : x.replace(' ', '_').lower(), entity_set))
    entity_set = set(entity_set)
    entities = pd.Series(list(entity_set))

    return data, entities


def getEntityFrequency():

    global data, entities

    _ = preprocess(True)
    _ = readSupplyData('div10000-aa.json')

    ec = pd.DataFrame(entities)
    ec['count'] = 0
    ec = ec.set_index(0)
    
    for index, value in data['title_entity'].iteritems():
        for i in value:
            ec.loc[i.replace(' ', '_').lower()]['count'] += 1


    ec.to_json('ec.json')


def createFile(path):
    _ = preprocess(True)
    _ = reformatFile('div10000-aa')
    _ = readSupplyData('div10000-aa.json')
    _ = addDescription('maps.txt')
    _ = saveForWord2Vec(path)


def test():

    global data

    _ = preprocess(True)
    _ = readSupplyData('div10000-aa.json')
    _ = addDescription('maps.txt')
    # saveForWord2Vec(path)
    _ = readVectors('w2v_vec')
    t = getDocumentData()
    for i in range(len(data)):
        print(i)
        _ = next(t)

# python data_gen.py data_train/articles_trainset.csv data_train/reformat_trainset.json data_train/entities_trainset.csv data_train/maps.txt word2vec-master/word2vec data_train/w2v_input data_train/w2v_vec
# python data_gen.py data_test/articles_testset.csv data_test/reformat_testset.json data_test/entities_testset.csv data_test/maps.txt word2vec-master/word2vec data_test/w2v_input data_test/w2v_vec
# python data_gen.py data_all/articles_all.csv data_all/reformat_all.json data_all/entities_all.csv data_all/maps.txt word2vec-master/word2vec data_all/w2v_input data_all/w2v_vec
def data_prepare(art_path, entity_reform_path, entity_path, description_path, w2v_dir, w2v_input_path, w2v_vec_path):
    global data, entities

    # preprocess
    print('preprocess')
    data = pd.read_csv(art_path, keep_default_na=False)
    data = data[['paperAbstract', 'title', 'id']]
    # reformatFile
    print('reformatFile')
    entity_data = pd.read_csv(entity_path, keep_default_na=False)
    result = []
    for index, erow in entity_data.iterrows():
        result.append([erow['id'], erow['title'].strip().split(','), erow['paperAbstract'].strip().split(',')])
    result = pd.DataFrame(result)
    result = result.set_index(0)
    result.to_json(entity_reform_path)
    # readSupplyData
    print('readSupplyData')
    readSupplyData(entity_reform_path)
    # addDescription
    print('addDescription')
    addDescription(description_path)
    # saveForWord2Vec
    print('saveForWord2Vec')
    saveForWord2Vec(w2v_input_path)
    os.system('time %s -train %s -output %s -cbow 0 -size 128 -window 8 -negative 25 '
              '-hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15 -min-count 0' % (w2v_dir, w2v_input_path, w2v_vec_path))


if __name__ == '__main__':
    arglis = sys.argv[1:]
    arglis = tuple(arglis)
    data_prepare(*arglis)
