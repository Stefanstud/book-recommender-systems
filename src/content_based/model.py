import torch
import torch.nn as nn


class ContentBasedModel(nn.Module):
    def __init__(self, input_dim, author_dim, category_dim, embedding_dim=50):
        super(ContentBasedModel, self).__init__()
        self.text_fc = nn.Linear(input_dim, embedding_dim)
        self.author_embedding = nn.Embedding(author_dim, embedding_dim)
        self.category_embedding = nn.Embedding(category_dim, embedding_dim)
        self.category_fc = nn.Linear(embedding_dim * 10, embedding_dim)
        self.page_count_fc = nn.Linear(1, embedding_dim)
        self.relu = nn.ReLU()
        self.output_fc = nn.Linear(embedding_dim * 4, 1)

    def forward(self, text_features, author_id, category_ids, page_count):
        text_out = self.relu(
            self.text_fc(text_features)
        ) 

        author_out = self.author_embedding(
            author_id
        ) 

        category_embeds = self.category_embedding(
            category_ids
        ) 
        category_out = self.relu(
            self.category_fc(category_embeds.view(category_embeds.size(0), -1))
        ) 

        page_count_out = self.relu(
            self.page_count_fc(page_count.unsqueeze(1))
        )

        combined_features = torch.cat(
            [text_out, author_out, category_out, page_count_out], dim=1
        )

        output = self.output_fc(combined_features).squeeze(1) 
        return output
