import torch
import pandas as pd

class Loader(torch.utils.data.Dataset):
    def __init__(self, df, year_intervals, age_intervals):
        self.ratings = df.copy()
        mappings = {
            'usr2idx': 'User-ID',
            'book2idx': 'ISBN',
            'loc2idx': 'Location',
            'auth2idx': 'Book-Author',
        }

        for attr, col in mappings.items():
            unique_vals = df[col].unique()
            setattr(self, attr, {val: i for i, val in enumerate(unique_vals)})

        self.idx2usr = {u: i for i, u in self.usr2idx.items()}
        self.idx2book = {b: i for i, b in self.book2idx.items()} #inner to isbn
        self.idx2auth = {a: i for i, a in self.auth2idx.items()}

        self.ratings['ISBN'] = df['ISBN'].apply(lambda x: self.book2idx[x])
        self.ratings['User-ID'] = df['User-ID'].apply(lambda x: self.usr2idx[x])

        self.ratings['Book-Author'] = df['Book-Author'].apply(lambda x: self.auth2idx[x])
        self.ratings['Location'] = df['Location'].apply(lambda x: self.loc2idx[x])

        self.ratings['Year-Of-Publication'] = pd.cut(
            df['Year-Of-Publication'],
            bins=year_intervals,
            labels=False,
            include_lowest=True
        ).fillna(0).astype(int)

        self.ratings['Age'] = pd.cut(
            df['Age'],
            bins=age_intervals,
            labels=False,
            include_lowest=True
        ).fillna(0).astype(int)

        self.x = self.ratings[[
            'ISBN',
            'User-ID',
            'Book-Author', 'Year-Of-Publication',
            'Location', 'Age'
        ]].to_numpy(dtype=int)

        self.y = self.ratings['Book-Rating'].to_numpy(float)

        self.x = torch.tensor(self.x, dtype=torch.long)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.ratings)


class Recommender(torch.nn.Module): # neural collaborative with feature embeddings
    def __init__(self, n_users, n_items, n_umeta, n_imeta, hidden_dim=64, hidden_meta_dim=8, dropout=0.3):
        super().__init__()

        # embeddings
        self.user_embedding = torch.nn.Embedding(n_users, hidden_dim)
        self.item_embedding = torch.nn.Embedding(n_items, hidden_dim)
        self.user_meta_embedding = torch.nn.Embedding(n_umeta, hidden_meta_dim)
        self.item_meta_embedding = torch.nn.Embedding(n_imeta, hidden_meta_dim)

        # bias terms
        self.user_bias = torch.nn.Embedding(n_users, 1)
        self.item_bias = torch.nn.Embedding(n_items, 1)
        self.global_bias = torch.nn.Parameter(torch.zeros(1))

        # head
        self.head = torch.nn.Sequential(
            # torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.Linear(hidden_dim * 2 + (hidden_meta_dim * 2) * 2, hidden_dim + (hidden_meta_dim * 2)),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            # torch.nn.LazyLinear(hidden_dim // 2),
            torch.nn.LazyLinear(hidden_dim + (hidden_meta_dim * 2) // 2),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(1)
        )

    def forward(self, x):
        bsz = x.shape[0]
        item_encoded, user_encoded = x[:, 0], x[:, 1]
        item_meta_encoded = x[:, 2:4]
        user_meta_encoded = x[:, 4:6]

        # embeddings
        user_emb = self.user_embedding(user_encoded)
        item_emb = self.item_embedding(item_encoded)
        user_meta_emb = self.user_meta_embedding(user_meta_encoded).view(bsz, -1)
        item_meta_emb = self.item_meta_embedding(item_meta_encoded).view(bsz, -1)

        # combine embeddings
        user_hidden = torch.cat((user_emb, user_meta_emb), dim=1)
        item_hidden = torch.cat((item_emb, item_meta_emb), dim=1)
        hidden = torch.cat((user_hidden, item_hidden), dim=1)

        out = self.head(hidden).squeeze(-1)

        # prediction = out + user_b + item_b + self.global_bias
        return out


