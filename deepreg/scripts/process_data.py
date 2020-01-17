import json
import numpy

import pandas

import os

import pickle
import itertools

from sklearn import preprocessing

from deepreg.datasets.calls_dataset import CallsDataset_V2
from deepreg.tools.config import EOD, BOD, PAD, SPECIAL_TOKEN_IDS
from deepreg.tools.misc import prettyformat_dict_string



### Preliminaries

BASE_PATH = '.'

#data_source_version = 4
#data_preprocessing_version = 6

calls_input_dir = BASE_PATH + '/data/source_data/transcripts'
data_input_dir = BASE_PATH + '/data/source_data/financials'
data_output_dir = BASE_PATH + '/data/transformed_data'



# Step 1: join finance features and text data
#
# Save data in a format that can be read fast:
#
# data file (joined_data_dat_file), each new line contains a pickled call
# index file (joined_data_loc_file), holds a list, each entry is a byte-offset of line


### Set up some variable names and also set some options

# Options

# do_normalize = True
do_normalize = False

# scaler = preprocessing.StandardScaler()
scaler = preprocessing.MinMaxScaler()


joined_data_dat_file = 'data_joined.dat'
joined_data_loc_file = 'data_joined.loc'
joined_data_text_stats_file = 'data_joined.stats'

joined_padded_data = 'data_padded.dat'
joined_padded_loc = 'data_padded.loc'
joined_padded_stats = 'data_padded.stats'

dataset_files = ['train', 'validate', 'test', ]

id_column_name = ['ID']
features_column_name = ['VIX', 'SIZE', 'VOLA_PRIOR', 'BTM', 'INDUSTRY', 'SUE']
features_column_name_to_normalize = ['VIX', 'SIZE', 'VOLA_PRIOR', 'VOLA_AFTER', 'BTM', 'SUE']
features_column_name_categorial = ['INDUSTRY']
target_column_name = ['VOLA_AFTER']


os.makedirs(data_input_dir, exist_ok=True)
os.makedirs(data_output_dir, exist_ok=True)

for split, dataset_file in enumerate(dataset_files):

    if not os.path.exists(os.path.join(data_output_dir, '{}_{}'.format(dataset_file, joined_data_dat_file))):

        df = pandas.read_csv(data_input_dir + '/{}.csv'.format(dataset_file))

        print(df.columns)

        df['SIZE'] = df['SIZE'].apply(numpy.log)

        if do_normalize:
            x = df[features_column_name_to_normalize].values
            if split == 0:
                x_scaled = scaler.fit_transform(x)
            else:
                x_scaled = scaler.transform(x)

            df[features_column_name_to_normalize] = x_scaled

        df_target = df[target_column_name]
        df_features = df[features_column_name]
        df_id = df[id_column_name]

        df_features = pandas.get_dummies(df_features, columns=features_column_name_categorial)

        joined_data_loc_list = list()
        with open(os.path.join(data_output_dir, '{}_{}'.format(dataset_file, joined_data_dat_file)), 'wb') as f:
            for i, (indexed_call_id, features, vola_after) in enumerate(zip(df_id.itertuples(), df_features.itertuples(), df_target.itertuples())):
                indexed_call_id, features, vola_after = indexed_call_id[:][1], features[1:], vola_after[:][1]
                if i == 0:
                    print(indexed_call_id, vola_after, features)
                if os.path.exists(os.path.join(calls_input_dir, indexed_call_id)):
                    indexed_call_array = json.load(open(os.path.join(calls_input_dir, indexed_call_id)))
                    if len(indexed_call_array) > 0:
                        joined_data_loc_list.append(str(f.tell()) + '\n')  # remember row byte offset
                        pickle.dump((indexed_call_id, vola_after, features, indexed_call_array), f)  # write new row
                # if i == 100:
                #     break

        with open(os.path.join(data_output_dir, '{}_{}'.format(dataset_file, joined_data_loc_file)), 'w') as f_loc:
            f_loc.writelines(joined_data_loc_list)



# Step 2: Compute statistics about sentence lengths and token per sentence lengths
# Will be used as feature and for computing the cutoffs and padding


presentation_sen_len = list()
presentation_tok_len = list()

questions_answers_len = list()

questions_sen_len = list()
questions_tok_len = list()

answers_sen_len = list()
answers_tok_len = list()

if not os.path.exists(os.path.join(data_output_dir, joined_data_text_stats_file)):

    with open(os.path.join(data_output_dir, '{}_{}'.format('train', joined_data_dat_file)), 'rb') as f:
        line = pickle.load(f)
        while line:
            indexed_call_id, vola_after, features, indexed_call_array = line
            if len(indexed_call_array) > 1:

                presentation_sens = list(itertools.chain.from_iterable(indexed_call_array[0]))
                presentation_sen_len.append(len(presentation_sens))
                presentation_tok_len.extend(map(len, presentation_sens))

                questions_answers_len.append(len(indexed_call_array[1][0::2]))

#                if data_preprocessing_version >= 3:
                questions_sens = list(itertools.chain.from_iterable(indexed_call_array[1][0::2]))
                questions_sen_len.append(len(questions_sens))
                questions_tok_len.extend(map(len, questions_sens))

                answers_sens = list(itertools.chain.from_iterable(indexed_call_array[1][1::2]))
                answers_sen_len.append(len(answers_sens))
                answers_tok_len.extend(map(len, answers_sens))

            try:
                line = pickle.load(f)
            except EOFError:
                break

        joined_data_text_stats = {
            'presentation_sen_len': pandas.DataFrame(presentation_sen_len).describe(percentiles=[.25, .5, .75, .80, .85, .90, .95]).to_dict(),
            'presentation_tok_len': pandas.DataFrame(presentation_tok_len).describe(percentiles=[.25, .5, .75, .80, .85, .90, .95]).to_dict(),
            'questions_answers_len': pandas.DataFrame(questions_answers_len).describe(percentiles=[.25, .5, .75, .80, .85, .90, .95]).to_dict(),
            'questions_sen_len': pandas.DataFrame(questions_sen_len).describe(percentiles=[.25, .5, .75, .80, .85, .90, .95]).to_dict(),
            'questions_tok_len': pandas.DataFrame(questions_tok_len).describe(percentiles=[.25, .5, .75, .80, .85, .90, .95]).to_dict(),
            'answers_sen_len': pandas.DataFrame(answers_sen_len).describe(percentiles=[.25, .5, .75, .80, .85, .90, .95]).to_dict(),
            'answers_tok_len': pandas.DataFrame(answers_tok_len).describe(percentiles=[.25, .5, .75, .80, .85, .90, .95]).to_dict(),
        }

        with open(os.path.join(data_output_dir, joined_data_text_stats_file), 'wb') as f:
            pickle.dump(joined_data_text_stats, f)

with open(os.path.join(data_output_dir, joined_data_text_stats_file), 'rb') as f:
    joined_data_text_stats = pickle.load(f)

print(prettyformat_dict_string(joined_data_text_stats))



# Step 3: Create padded and truncated sequences + text features + finance features
# and write into file


# Define helper function
def limit_pad_document(
        input_sentences,
        max_sen_tok,
        max_sen=1,
):
    if len(input_sentences) == 0:
        input_sentences = [PAD]
    if isinstance(max_sen_tok, float):
        max_sen_tok = int(max_sen_tok)
    assert max_sen_tok > 0, 'max_sen_tok <= 0'
    if not isinstance(input_sentences[0], list):
        input_sentences = [input_sentences]
    sents = numpy.zeros((max_sen, max_sen_tok + 2))
    for sid, sentence in enumerate(input_sentences):
        # print(len(sentence[:max_sen_tok]))
        sentence = numpy.array(sentence[:max_sen_tok]) + len(SPECIAL_TOKEN_IDS)  # cutoff sentence length and increase embedding id
        sents[sid][1:len(sentence) + 1] = sentence  # copy ids
        sents[sid][0], sents[sid][len(sentence) + 1] = BOD, EOD  # wrap with begin/end-of-sequence (BOS) and (EOS)
    return sents


for split, dataset_file in enumerate(dataset_files):

    max_id = 0
    feature_size = 0

    if not os.path.exists(os.path.join(data_output_dir, '{}_{}'.format(dataset_file, joined_padded_data))):
        joined_padded_data_loc_list = list()
        with open(os.path.join(data_output_dir, '{}_{}'.format(dataset_file, joined_data_dat_file)), 'rb') as f:
            with open(os.path.join(data_output_dir, '{}_{}'.format(dataset_file, joined_padded_data)), 'wb') as f_out:
                line = pickle.load(f)
                while line:
                    indexed_call_id, vola_after, features, indexed_call_array = line

                    presentation_sens = list(itertools.chain.from_iterable(indexed_call_array[0]))
                    presentation_toks = list(itertools.chain.from_iterable(presentation_sens))

                    question_1_toks = [[PAD]]
                    answer_1_toks = [[PAD]]
                    question_2_toks = [[PAD]]
                    answer_2_toks = [[PAD]]

                    if len(indexed_call_array) > 1:
                        #if data_preprocessing_version >= 3:
                        question_1_toks = list(itertools.chain.from_iterable(itertools.chain.from_iterable(indexed_call_array[1][0::2])))
                        answer_1_toks = list(itertools.chain.from_iterable(itertools.chain.from_iterable(indexed_call_array[1][1::2])))

                    presentation_percentile = '25%' # presentations where mostly long
                    qa_percentile = '90%' # qas where mostly short

                    presentation_len = joined_data_text_stats['presentation_sen_len'][0][presentation_percentile] * joined_data_text_stats['presentation_tok_len'][0][presentation_percentile]
                    question_len = joined_data_text_stats['questions_sen_len'][0][presentation_percentile] * joined_data_text_stats['questions_tok_len'][0][presentation_percentile]
                    answer_len = joined_data_text_stats['answers_sen_len'][0][presentation_percentile] * joined_data_text_stats['answers_tok_len'][0][presentation_percentile]

                    presentation_toks_np = limit_pad_document(
                        presentation_toks,
                        presentation_len
                    )

                    # Use the
                    question_1_toks_np = limit_pad_document(
                        question_1_toks,
                        presentation_len
                    )
                    answer_1_toks_np = limit_pad_document(
                        answer_1_toks,
                        presentation_len
                    )

                    max_id = max(max_id, numpy.max(presentation_toks_np))
                    max_id = max(max_id, numpy.max(question_1_toks_np))
                    max_id = max(max_id, numpy.max(answer_1_toks_np))

                    presentation_sen_len_norm = (len(presentation_sens) - joined_data_text_stats['presentation_sen_len'][0]['mean']) / joined_data_text_stats['presentation_sen_len'][0]['std']
                    presentation_tok_len_norm = (numpy.mean(list(map(len, presentation_sens))) - joined_data_text_stats['presentation_tok_len'][0]['mean']) / joined_data_text_stats['presentation_tok_len'][0]['std']

                    questions_answers_len.append(len(indexed_call_array[0][1::2]))

                    if len(indexed_call_array[0]) > 2:

                        question_sens = list(itertools.chain.from_iterable(indexed_call_array[0][1::2]))
                        questions_sen_len_norm = (numpy.mean(list(map(len, indexed_call_array[0][1::2]))) - joined_data_text_stats['questions_sen_len'][0]['mean']) / joined_data_text_stats['questions_sen_len'][0]['std']
                        questions_tok_len_norm = (numpy.mean(list(map(len, itertools.chain.from_iterable(indexed_call_array[0][1::2])))) - joined_data_text_stats['questions_tok_len'][0]['mean']) / joined_data_text_stats['questions_tok_len'][0]['std']

                        answers_sens = list(itertools.chain.from_iterable(indexed_call_array[0][2::2]))
                        answers_sen_len_norm = (numpy.mean(list(map(len, indexed_call_array[0][2::2]))) - joined_data_text_stats['answers_sen_len'][0]['mean']) / joined_data_text_stats['answers_sen_len'][0]['std']
                        answers_tok_len_norm = (numpy.mean(list(map(len, itertools.chain.from_iterable(indexed_call_array[0][2::2])))) - joined_data_text_stats['answers_tok_len'][0]['mean']) / joined_data_text_stats['answers_tok_len'][0]['std']

                    else:

                        questions_sen_len_norm = 0
                        questions_tok_len_norm = 0

                        answers_sen_len_norm = 0
                        answers_tok_len_norm = 0

                    joined_padded_data_loc_list.append(str(f_out.tell()) + '\n')  # record row byte offset
                    pickle.dump((
                        indexed_call_id,
                        vola_after,
                        features + tuple([
                            len(indexed_call_array[0][1:]) / 2,  # how many questions/answers
                            presentation_sen_len_norm,
                            presentation_tok_len_norm,
                            questions_sen_len_norm,
                            questions_tok_len_norm,
                            answers_sen_len_norm,
                            answers_tok_len_norm
                        ]),
                        presentation_toks_np,
                        question_1_toks_np,
                        answer_1_toks_np,
                    ), f_out)  # writeout feature row
                    feature_size = len(features) + 7  # finance feature size and text features

                    try:
                        line = pickle.load(f)
                    except EOFError:
                        break

        with open(os.path.join(data_output_dir, '{}_{}'.format(dataset_file, joined_padded_loc)), 'w') as f_loc:
            f_loc.writelines(joined_padded_data_loc_list)

        joined_padded_stats_dict = {
            'max_id': max_id,
            'feature_size': feature_size,
        }

        with open(os.path.join(data_output_dir, '{}_{}'.format(dataset_file, joined_padded_stats)), 'wb') as f:
            pickle.dump(joined_padded_stats_dict, f)

        print(joined_padded_stats_dict)

"""


with open(os.path.join(data_output_dir, '{}_{}'.format('train', joined_padded_data)), 'rb') as f:
    line = pickle.load(f)
    indexed_call_id, \
    vola_after, \
    features, \
    presentation_toks_np, \
    question_1_toks_np, \
    answer_1_toks_np = line
    print(line)

    #question_2_toks_np, \
    #answer_2_toks_np = line


calls_dataset = CallsDataset_V2(
    dataset_dir=data_output_dir,
    data_padded_dat='{}_{}'.format('train', joined_padded_data),
    data_padded_loc='{}_{}'.format('train', joined_padded_loc),
    data_padded_stats='{}_{}'.format('train', joined_padded_stats),
)

calls_dataset_loader = calls_dataset.get_loader(
    batch_size=16,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

for bid, batch in enumerate(calls_dataset_loader):
    print(bid, len(batch))
    print(0, batch[0])
    print(1, batch[1])
    print(2, batch[2])
    print(3, batch[3])
    print(4, batch[4])
    print(5, batch[5])
    print(6, batch[6])
    break

"""
