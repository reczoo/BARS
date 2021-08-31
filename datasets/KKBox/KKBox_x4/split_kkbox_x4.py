import numpy as np
import shutil
import pandas as pd
import os
import json
import re
from sklearn.model_selection import StratifiedKFold


RANDOM_SEED = 2018 # Set seed for reproduction
datapath = "./kkbox-music-recommendation-challenge/"

# !!! Directly using pd.read_csv() leads an error: #rows < 2296833
# songs_df = pd.read_csv(os.path.join(datapath, "songs.csv"), encoding="utf-8", dtype=object)
song_list = []
song_header = []
with open(os.path.join(datapath, "songs.csv"), 'r', encoding="utf-8") as fid:
    k = 0
    for line in fid:
        k += 1
        splits = line.strip().split(",")
        if len(splits) != 7:
            print(line)
            splits = splits[0:7] # correction
        if k == 1:
            print("headers", splits)
            song_header = splits
        else:
            song_list.append(splits)
songs_df = pd.DataFrame(song_list, columns=song_header)
print("songs_df shape", songs_df.shape)
songs_df['language'] = songs_df['language'].map(lambda x: str(int(float(x))) if not pd.isnull(x) else "")
songs_df['genre_ids'] = songs_df['genre_ids'].map(lambda x: x.replace("|", " ") if not pd.isnull(x) else "")
song_ids = set(songs_df['song_id'].dropna().unique())

person_names = set(songs_df['artist_name'].dropna().unique()) | set(songs_df['composer'].dropna().unique())\
               | set(songs_df['lyricist'].dropna().unique())

def name_tokenize(name_str):
    persons = re.split(r"[\|\\/&;]", name_str)
    return [x.replace("\"", "").strip() for x in persons if x.replace("\"", "").strip() != ""]

person_set = []
for name_str in person_names:
    person_set += name_tokenize(name_str)
person_set = set(person_set)
person_set = sorted(list(person_set)) # sort for reproduction
person_dict = dict(list(zip(person_set, range(1, len(person_set) + 1))))
with open("person_id.json", "w", encoding="utf-8") as fout:
    person_index = dict(list(zip(range(1, len(person_set) + 1), person_set)))
    json.dump(person_index, fout, indent=4, ensure_ascii=False)
    del person_index

def encode_name(name_str):
    names = name_tokenize(name_str)
    names = [str(person_dict[x]) for x in names]
    return " ".join(names)

songs_df['artist_name'] = songs_df['artist_name'].map(lambda x: encode_name(x) if not pd.isnull(x) else "")
songs_df['composer'] = songs_df['composer'].map(lambda x: encode_name(x) if not pd.isnull(x) else "")
songs_df['lyricist'] = songs_df['lyricist'].map(lambda x: encode_name(x) if not pd.isnull(x) else "")

# !!! Directly using pd.read_csv() leads an error: #rows < 2296869
# song_extra_info_df = pd.read_csv(os.path.join(datapath, "song_extra_info.csv"), encoding="utf-8")
song_extra_list = []
song_extra_header = []
with open(os.path.join(datapath, "song_extra_info.csv"), 'r', encoding="utf-8") as fid:
    k = 0
    for line in fid:
        k += 1
        splits = line.strip().split(",")
        if len(splits) != 3:
            print(line)
        if k == 1:
            song_extra_header = splits
        else:
            song_extra_list.append(splits)
    print(k - 1, "lines in song_extra_info.csv")
song_extra_info_df = pd.DataFrame(song_extra_list, columns=song_extra_header)
print("song_extra_info_df shape", song_extra_info_df.shape)
song_ids = song_ids | set(song_extra_info_df['song_id'].dropna().unique())
song_names = set(song_extra_info_df['name'].dropna().unique())
song_names = sorted(list(song_names))
song_name_dict = dict(list(zip(song_names, range(1, len(song_names) + 1))))
song_extra_info_df["name"] = song_extra_info_df["name"].map(lambda x: song_name_dict[x] if not pd.isnull(x) else "")
with open("song_name.json", "w", encoding="utf-8") as fout:
    song_name_index = dict(list(zip(range(1, len(song_names) + 1), song_names)))
    json.dump(song_name_index, fout, indent=4, ensure_ascii=False)
    del song_name_index

with open(os.path.join(datapath, "members.csv"), 'r') as fid:
    print(sum(1 for line in fid) - 1, "lines in members.csv")
members_df = pd.read_csv(os.path.join(datapath, "members.csv"))
print("members_df shape", members_df.shape)
user_ids = set(members_df['msno'].dropna().unique())

with open(os.path.join(datapath, "train.csv"), 'r') as fid:
    print(sum(1 for line in fid) - 1, "lines in train.csv")
train_df = pd.read_csv(os.path.join(datapath, "train.csv"))
print("train_df shape", train_df.shape)
song_ids = sorted(list(song_ids | set(train_df['song_id'].dropna().unique())))
user_ids = sorted(list(user_ids | set(train_df['msno'].dropna().unique())))
song_dict = dict(list(zip(song_ids, range(1, len(song_ids) + 1))))
user_dict = dict(list(zip(user_ids, range(1, len(user_ids) + 1))))
with open("user_id.json", "w") as fout:
    user_index = dict(list(zip(range(1, len(user_ids) + 1), user_ids)))
    json.dump(user_index, fout, indent=4)
    del user_index
with open("song_id.json", "w") as fout:
    song_index = dict(list(zip(range(1, len(song_ids) + 1), song_ids)))
    json.dump(song_index, fout, indent=4)
    del song_index

train_with_user = pd.merge(train_df, right=members_df, on="msno", how="left")
train_with_user_song = pd.merge(train_with_user, right=songs_df, on="song_id", how="left")
train_with_user_song_extra = pd.merge(train_with_user_song, right=song_extra_info_df, on="song_id", how="left")
print("columns are", train_with_user_song_extra.columns)

train_with_user_song_extra['label'] = train_with_user_song_extra['target'].map(lambda x: "1" if float(x) > 0 else "0")
ddf = train_with_user_song_extra.reindex(columns=[u'label', u'msno', u'song_id', u'source_system_tab', u'source_screen_name',
                                                 u'source_type', u'city', u'bd', u'gender', u'registered_via',
                                                 u'registration_init_time', u'expiration_date', u'song_length', u'genre_ids', 
                                                 u'artist_name', u'composer', u'lyricist', u'language',
                                                 u'name', u'isrc'])
ddf[u'msno'] = ddf[u'msno'].map(lambda x: user_dict[x] if not pd.isnull(x) else "")
ddf[u'song_id'] = ddf[u'song_id'].map(lambda x: song_dict[x] if not pd.isnull(x) else "")
print("ddf shape", ddf.shape)

X = ddf.values
y = ddf['label'].map(lambda x: float(x)).values
print(str(len(X)) + ' lines in total')

folds = StratifiedKFold(n_splits=10, shuffle=True,
                        random_state=RANDOM_SEED).split(X, y)

fold_indexes = []
for train_id, valid_id in folds:
    fold_indexes.append(valid_id)
test_index = fold_indexes[0]
valid_index = fold_indexes[1]
train_index = np.concatenate(fold_indexes[2:])

test_df = ddf.loc[test_index, :]
test_df.to_csv('test.csv', index=False, encoding='utf-8')
valid_df = ddf.loc[valid_index, :]
valid_df.to_csv('valid.csv', index=False, encoding='utf-8')
ddf.loc[train_index, :].to_csv('train.csv', index=False, encoding='utf-8')

print('Train lines:', len(train_index))
print('Validation lines:', len(valid_index))
print('Test lines:', len(test_index))
print('Postive samples:', np.sum(y))
print('Postive ratio:', np.sum(y) / len(y))
print('Postive test:', np.sum(test_df['label'].map(lambda x: float(x))))
print('Postive validation:', np.sum(valid_df['label'].map(lambda x: float(x))))

