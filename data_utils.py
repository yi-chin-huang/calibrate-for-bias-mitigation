import pandas as pd
import json
import pickle
import numpy as np
from utils import ROOT_DIR

def load_eec_test():
    with open(f"{ROOT_DIR}/data/Equity-Evaluation-Corpus.csv", "r") as f:
        lines = f.readlines()
        sentences = []
        templates = []
        emotion_words = []
        genders = []
        races = []
        for idx, new_line in enumerate(lines):
            if idx == 0:
                continue
            line = new_line.split(',')
            if line[6] == '':
                continue
            sentences.append(line[1])
            templates.append(line[2])
            emotion_words.append(line[6])
            genders.append(line[4])
            races.append(line[5])
        # print('---sentences---')
        # print(sentences)
        # print("---templates---")
        # print(templates)
        # print("---emotion_words---")
        # print(emotion_words)
        # print("---genders---")
        # print(genders)
        # print("---races---")
        # print(races)
        return sentences, templates, emotion_words, genders, races
    # orig_train_sentences = ["The person feels angry.", "The person feels furious"]
    # orig_train_labels = [1, 2]
    # orig_test_sentences = ["The conversation with this boy was outrageous.", "I made Ebony feel irritated.", "The situation makes Latisha feel enraged."]
    # orig_test_labels = [3, 0, 4]
    # return sentences, persons, genders, races

def load_neutral_input():
    sentences = []
    race_template = ["%s", "An %s", "An %s person", "The %s person"]
    gender_template = ["%s", "A %s", "The %s"]
    
    for i, template in enumerate(race_template):
        sentences.append(template % 'African-American')
        sentences.append(template % 'European')
    for i, template in enumerate(gender_template):
        sentences.append(template % 'man')
        sentences.append(template % 'woman')
        sentences.append(template % 'boy')
        sentences.append(template % 'girl')
        sentences.append(template % 'male')
        sentences.append(template % 'female')


    sentence_cnt = len(sentences)
    emotion_words = ['anger'] * sentence_cnt + ['joy'] * sentence_cnt + ['fear'] * sentence_cnt + ['sadness'] * sentence_cnt
    expanded_sentences = sentences + sentences + sentences + sentences
    temp_list = ['' for i in range(len(expanded_sentences))]
    
    return expanded_sentences, temp_list, emotion_words, temp_list, temp_list

def load_dataset(params):
    """
    Load train and test data
    :param params: experiment parameter, which contains dataset spec
    :return: train_x, train_y, test_x, test_y
    """

    if params['dataset'] == 'eec':
        sentences, templates, emotion_words, genders, races = load_neutral_input()
        params['prompt_prefix'] = "From scores of one, two, three, four, five, classify the intensity of %s that this sentence conveys.\n\n"
        params["q_prefix"] = "Sentence: "
        params["a_prefix"] = "Emotional Intensity: "
        params['label_dict'] = {0: ['one'], 1: ['two'], 2: ['three'], 3: ['four'], 4: ['five']}
        params['inv_label_dict'] = {' one': 0, ' two': 1, ' three': 2, ' four': 3, ' five': 4}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    else:
        raise NotImplementedError
    return sentences, templates, emotion_words, genders, races