import torch
import numpy as np
import pandas as pd
import pickle as pkl
import torch.nn as nn

import config
from train import load_model

user_job = {0: 'other or not specified', 1: 'academic/educator', 2: 'artist',
            3: 'clerical/admin', 4: 'college/grad student', 5: 'customer service',
            6: 'doctor/health care', 7: 'executive/managerial', 8: 'farmer',
            9: 'homemaker', 10: 'K-12 student', 11: 'lawyer', 12: 'programmer',
            13: 'retired', 14: 'sales/marketing', 15: 'scientist', 16: 'self-employed',
            17: 'technician/engineer', 18: 'tradesman/craftsman', 19: 'unemployed',
            20: 'writer'
            }
user_gender = {'F': 'Female', 'M': 'Male'}
user_age = {
    1: 'Under 18',
    18: '18-24',
    25: '25-34',
    35: '35-44',
    45: '45-49',
    50: '50-55',
    56: '56+'
}


def load_preprocessed_data(args):
    users_raw = pkl.load(open(args.path + 'users_raw.pkl', 'rb'))
    movies_raw = pkl.load(open(args.path + 'movies_raw.pkl', 'rb'))
    user_index_to_uid = pkl.load(open(args.path + 'user_index_to_uid.pkl', 'rb'))
    movie_index_to_mid = pkl.load(open(args.path + 'movie_index_to_mid.pkl', 'rb'))
    
    return users_raw, movies_raw, user_index_to_uid, movie_index_to_mid


def saveRankMatrix(args, model, train_iter):
    """
    得到用户特征矩阵与电影特征矩阵并计算得到评分矩阵 (m * n)
    """
    _, _, user_index_to_uid, movie_index_to_mid = load_preprocessed_data(args)
    user_feature_dict = {}
    movie_feature_dict = {}
    with torch.no_grad():
        for i_batch, sample_batch in enumerate(train_iter):
            user_inputs = sample_batch['user_inputs']
            movie_inputs = sample_batch['movie_inputs']
            
            # [batch_size, hidden_dim]
            _, user_feature, movie_feature = model(user_inputs, movie_inputs)
            user_feature = user_feature.cpu().numpy()
            movie_feature = movie_feature.cpu().numpy()
            
            for i in range(user_inputs['uid'].shape[0]):
                uid = user_inputs['uid'][i]  # uid
                mid = movie_inputs['mid'][i]  # mid
                
                if uid.item() not in user_feature_dict.keys():
                    user_feature_dict[uid.item()] = user_feature[i]
                if mid.item() not in movie_feature_dict.keys():
                    movie_feature_dict[mid.item()] = movie_feature[i]
            if i_batch % 100 == 0:
                print('Steps: {}\{}'.format(i_batch, len(train_iter)))
    
    # 按 ID 排序
    user_feature_dict = dict(sorted(user_feature_dict.items(), key=lambda ele: ele[0], reverse=False))
    movie_feature_dict = dict(sorted(movie_feature_dict.items(), key=lambda ele: ele[0], reverse=False))
    
    # 用户特征矩阵 (m * hidden_dim)
    user_feature_matrix = torch.FloatTensor(list(user_feature_dict.values())).to(args.device)
    # 电影特征矩阵 (n * hidden_dim)
    movie_feature_matrix = torch.FloatTensor(list(movie_feature_dict.values())).to(args.device)
    # 评分矩阵 (m * n)
    rank_matrix = torch.matmul(user_feature_matrix, movie_feature_matrix.T)
    rank_matrix = pd.DataFrame(rank_matrix.tolist())
    rank_matrix.columns = movie_index_to_mid
    rank_matrix.index = user_index_to_uid
    pkl.dump(rank_matrix, open('Params/rank_matrix.pkl', 'wb'))
    return rank_matrix


def evaluation_model(args, model, test_iter):
    losses = []
    criterion = nn.MSELoss()
    with torch.no_grad():
        for i_batch, sample_batch in enumerate(test_iter):
            user_inputs = sample_batch['user_inputs']
            movie_inputs = sample_batch['movie_inputs']
            target = torch.squeeze(sample_batch['target'].to(args.device))
            rank, _, _ = model(user_inputs, movie_inputs)
            loss = criterion(rank, target)
            losses.append(loss.item())
    return np.mean(losses)


def recommend_Movies(args, uid, rank_matrix):
    """
    根据 UserID 为其推荐电影与也喜欢看这些电影的用户 (相似用户)
    """
    users_raw, movies_raw, _, _ = load_preprocessed_data(args)
    user_ratings = rank_matrix.loc[uid].sort_values(ascending=False)
    top_k_movies = list(user_ratings.index)[0:args.recommend_num]
    similar_users = []
    print('该用户信息：' + 'UserID: {0}    Gender:{1}    Age:{2}    Occupation:{3}'.format(
        uid, user_gender[users_raw[uid]['Gender']], user_age[users_raw[uid]['Age']], user_job[users_raw[uid]['Job']]
    ))
    print('为该用户推荐的电影：')
    for i, mid in enumerate(top_k_movies):
        # 推荐电影
        movie_recommend = movies_raw[mid]
        print('{0}: MovieID: {1}    Movie Title: {2}    Movie Type: {3}'.format(
            i + 1, movie_recommend['Movie_ID'], movie_recommend['Movie_Title'], movie_recommend['Movie_Type']
        ))
        # 获取相似用户 (对推荐电影评分最高者)
        movie_ratings = rank_matrix[mid].sort_values(ascending=False)
        for j in range(len(movie_ratings)):
            similar_user = movie_ratings.index[j]
            if similar_user in similar_users:
                continue
            else:
                similar_users.append(similar_user)
                break
    print('为该用户推荐的相似用户：')
    for i, similar_user_id in enumerate(similar_users):
        user_recommend = users_raw[similar_user_id]
        print('{0}: UserID: {1}    Gender:{2}    Age:{3}    Occupation:{4}'.format(
            i + 1, similar_user_id, user_gender[user_recommend['Gender']],
            user_age[user_recommend['Age']], user_job[user_recommend['Job']]
        ))


def recommend_Users(args, mid, rank_matrix):
    """
    根据 MovieID 得到喜欢看该电影的用户与他们喜欢看的电影 (相似电影)
    """
    users_raw, movies_raw, _, _ = load_preprocessed_data(args)
    movie_ratings = rank_matrix[mid].sort_values(ascending=False)
    top_k_users = list(movie_ratings.index)[0:args.recommend_num]
    similar_movies = []
    print('该电影信息：' + 'MovieID: {0}    Movie Title: {1}    Movie Type: {2}'.format(
        mid, movies_raw[mid]['Movie_Title'], movies_raw[mid]['Movie_Type']
    ))
    print('喜欢看这部电影的用户：')
    for i, uid in enumerate(top_k_users):
        user_recommend = users_raw[uid]
        print('{0}: UserID: {1}    Gender:{2}    Age:{3}    Occupation:{4}'.format(
            i + 1, uid, user_gender[user_recommend['Gender']],
            user_age[user_recommend['Age']], user_job[user_recommend['Job']]
        ))
        user_ratings = rank_matrix.loc[uid].sort_values(ascending=False)
        for j in range(len(user_ratings)):
            similar_movie = user_ratings.index[j]
            if similar_movie in similar_movies:
                continue
            else:
                similar_movies.append(similar_movie)
                break
    print('他们也在看：')
    for i, similar_movie_id in enumerate(similar_movies):
        movie_recommend = movies_raw[similar_movie_id]
        print('{0}: MovieID: {1}    Movie Title: {2}    Movie Type: {3}'.format(
            i + 1, movie_recommend['Movie_ID'], movie_recommend['Movie_Title'], movie_recommend['Movie_Type']
        ))