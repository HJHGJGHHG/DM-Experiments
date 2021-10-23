import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


def get_movie_text_embedding(args):
    embedding = nn.Embedding(args.vocabulary_size, args.embedding_dim, device=args.device)
    if args.static:
        # embedding = embedding.from_pretrained(args.vectors, freeze=not args.non_static)
        embedding = embedding.from_pretrained(args.vectors, freeze=not args.non_static)
    
    return embedding


class MovieLens(nn.Module):
    def __init__(self, user_max_dict, movie_max_dict, args):
        super(MovieLens, self).__init__()
        self.args = args
        # --------------------------------- user channel ---------------------------------
        # user embeddings
        self.embedding_uid = nn.Embedding(user_max_dict['uid'], args.embedding_dim, device=args.device)
        
        # user info NN
        self.user_layer = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(
                args.embedding_dim + user_max_dict['gender'] + user_max_dict['age'] + user_max_dict['job'],
                args.hidden_dim, device=args.device),
            nn.ReLU(),
            
            nn.Linear(args.hidden_dim, args.hidden_dim, device=args.device),
            nn.Tanh(),
            nn.BatchNorm1d(args.hidden_dim, device=args.device)
        )
        
        # --------------------------------- movie channel ---------------------------------
        # movie embeddings
        self.embedding_mid = nn.Embedding(movie_max_dict['mid'], args.embedding_dim, device=args.device)  # normally 32
        
        # movie info NN
        self.movie_layer = nn.Sequential(
            nn.Linear(args.embedding_dim + movie_max_dict['mtype'], args.hidden_dim, device=args.device),
            nn.ReLU(),
            nn.BatchNorm1d(args.hidden_dim, device=args.device)
        )
        
        # movie title text
        self.text_embedding = get_movie_text_embedding(args)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, args.hidden_dim, (size, args.embedding_dim), device=args.device) for size in
             args.filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        
        self.movie_combine_layer = nn.Sequential(
            nn.Linear(args.hidden_dim + len(args.filter_sizes) * args.hidden_dim, args.hidden_dim, device=args.device),
            nn.Tanh()
        )
    
    def forward(self, user_input, movie_input):
        # --------------------------------- user channel ---------------------------------
        uid = torch.squeeze(
            self.embedding_uid(user_input['uid'].to(self.args.device)), dim=1)
        uid = torch.squeeze(uid, dim=1)  # [batch_size, embedding_dim]
        gender = torch.squeeze(user_input['gender'].to(self.args.device), dim=1)  # [batch_size, gender_types (2)]
        age = torch.squeeze(user_input['age'].to(self.args.device), dim=1)  # [batch_size, age_types (7)]
        job = torch.squeeze(user_input['job'].to(self.args.device), dim=1)  # [batch_size, job_types (21)]
        user_info = torch.cat([uid, gender, age, job],
                              dim=1)  # concat of user info tensors  [batch_size, sum of dimensions]
        user_feature = self.user_layer(user_info)  # user feature  [batch_size, hidden_dim]
        
        # --------------------------------- movie channel ---------------------------------
        mid = torch.squeeze(
            self.embedding_mid(movie_input['mid'].to(self.args.device)), dim=1)
        mid = torch.squeeze(mid, dim=1)  # [batch_size, embedding_dim]
        mtype = movie_input['mtype'].to(self.args.device)  # [batch_size, movie_types (18)]
        movie_info = torch.cat([mid, mtype], 1)  # concat of movie_id tensor and movie_type tensor
        movie_info = self.movie_layer(movie_info)  # movie info  [batch_size, hidden_dim]
        
        # movie title text
        mtext = movie_input['mtext'].to(self.args.device)  # [batch_size, seq_len (16)]
        text_embedding = self.text_embedding(mtext).unsqueeze(1)  # [batch_size, seq_len, text_embedding_dim]
        text_info = [F.relu(conv(text_embedding)).squeeze(3) for conv in self.convs]
        text_info = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in text_info]
        text_info = self.dropout(
            torch.cat(text_info, 1))  # movie text info  [batch_size, hidden_dim * len(filter_sizes)]
        
        movie_info = torch.cat([movie_info, text_info], dim=1)
        movie_feature = self.movie_combine_layer(movie_info)
        
        output = torch.sum(user_feature * movie_feature, 1)  # [batch_size, 1]
        
        return output, user_feature, movie_feature
