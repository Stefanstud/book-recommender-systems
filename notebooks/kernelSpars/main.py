import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import coo_matrix
from time import time
from tqdm import tqdm
import math 

# Set random seed for reproducibility
seed = int(time())
np.random.seed(seed)
torch.manual_seed(seed)

def loadData(path='./', valfrac=0.1, delimiter=',', seed=1234, transpose=False):
    np.random.seed(seed)
    
    print('Reading data...')
    data = pd.read_csv(path, delimiter=delimiter)
    print('Data read successfully.')
    user_ids = data['user_id'].unique()
    item_ids = data['book_id'].unique()
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    
    n_users = len(user_ids)
    n_items = len(item_ids)
    
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    val_size = int(valfrac * len(data))
    val_data = data[:val_size]
    train_data = data[val_size:]

    def create_sparse_matrix(df):
        row_indices = df['book_id'].map(item_id_to_index).values
        col_indices = df['user_id'].map(user_id_to_index).values
        values = df['rating'].values.astype('float32')
        return coo_matrix((values, (row_indices, col_indices)), shape=(n_items, n_users))
    
    tr_sparse = create_sparse_matrix(train_data)
    vr_sparse = create_sparse_matrix(val_data)
    
    if transpose:
        tr_sparse = tr_sparse.transpose()
        vr_sparse = vr_sparse.transpose()
    
    return tr_sparse, vr_sparse, user_id_to_index, item_id_to_index

tr_sparse, vr_sparse, user_id_to_index, item_id_to_index = loadData(
    '../../data/train.csv', delimiter=',', seed=seed, transpose=True, valfrac=0.1)

class KernelLayer(nn.Module):
    def __init__(self, n_in, n_hid=500, n_dim=5, activation=nn.Sigmoid(), lambda_s=0.006, lambda_2=20.0):
        super(KernelLayer, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_dim = n_dim
        self.activation = activation
        self.lambda_s = lambda_s
        self.lambda_2 = lambda_2
        
        self.W = nn.Parameter(torch.randn(n_in, n_hid))
        self.u = nn.Parameter(torch.randn(n_in, n_dim) * 1e-3)
        self.v = nn.Parameter(torch.randn(n_hid, n_dim) * 1e-3)
        self.b = nn.Parameter(torch.zeros(n_hid))
    
    def forward(self, x):
        w_hat = kernel(self.u, self.v)
        W_eff = self.W * w_hat
        y = torch.matmul(x, W_eff) + self.b
        y = self.activation(y)
        return y
    
    def regularization_loss(self):
        w_hat = kernel(self.u, self.v)
        sparse_reg_term = self.lambda_s * torch.norm(w_hat)
        l2_reg_term = self.lambda_2 * torch.norm(self.W)
        return sparse_reg_term + l2_reg_term

def kernel(u, v):
    dist = torch.cdist(u, v) ** 2
    hat = torch.clamp(1.0 - dist, min=0.0)
    return hat

class RecommendationModel(nn.Module):
    def __init__(self, n_in, n_hid=500, n_out=None, n_layers=2, n_dim=5, lambda_s=0.006, lambda_2=20.0):
        super(RecommendationModel, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(KernelLayer(n_in, n_hid, n_dim, activation=nn.Sigmoid(),
                                       lambda_s=lambda_s, lambda_2=lambda_2))
        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(KernelLayer(n_hid, n_hid, n_dim, activation=nn.Sigmoid(),
                                           lambda_s=lambda_s, lambda_2=lambda_2))
        # Output layer
        n_out = n_out if n_out is not None else n_in
        self.layers.append(KernelLayer(n_hid, n_out, n_dim, activation=nn.Identity(),
                                       lambda_s=lambda_s, lambda_2=lambda_2))
    
    def forward(self, x):
        reg_loss = 0
        for layer in self.layers:
            x = layer(x)
            reg_loss += layer.regularization_loss()
        return x, reg_loss

n_in = tr_sparse.shape[1]  
n_out = n_in  
n_hid = 500
lambda_s = 0.006
lambda_2 = 20.0
n_layers = 2
n_dim = 5
batch_size = 256
num_epochs = 10
learning_rate = 0.001

model = RecommendationModel(n_in, n_hid, n_out, n_layers, n_dim, lambda_s, lambda_2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

def generate_batches(sparse_matrix, batch_size):
    coo = sparse_matrix.tocoo()
    n_samples = len(coo.data)
    indices = np.vstack((coo.row, coo.col)).T
    values = coo.data
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        batch_values = values[start_idx:end_idx]
        batch_sparse = coo_matrix((batch_values, (batch_indices[:, 0], batch_indices[:, 1])),
                                  shape=sparse_matrix.shape)
        batch_dense = torch.FloatTensor(batch_sparse.toarray())
        yield batch_dense

model.train()
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    batch_generator = generate_batches(tr_sparse, batch_size)
    total_batches = int(np.ceil(tr_sparse.nnz / batch_size))
    epoch_loss = 0
    for batch_dense in tqdm(batch_generator, total=total_batches, desc="Training", unit="batch"):
        batch_dense = batch_dense.to(device)
        optimizer.zero_grad()
        outputs, reg_loss = model(batch_dense)
        mask = batch_dense != 0
        loss = criterion(outputs[mask], batch_dense[mask])
        total_loss = loss + reg_loss
        total_loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / total_batches
    print(f"Average Training Loss: {avg_loss}")
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_dense = torch.FloatTensor(vr_sparse.toarray()).to(device)
        outputs, reg_loss = model(val_dense)
        mask = val_dense != 0
        mse_loss = criterion(outputs[mask], val_dense[mask])
        rmse = math.sqrt(mse_loss.item())
        print(f"Validation Loss (MSE) after epoch {epoch + 1}: {mse_loss.item():.4f}")
        print(f"Validation RMSE after epoch {epoch + 1}: {rmse:.4f}")
    model.train()

# Evaluation on Validation Set
model.eval()
with torch.no_grad():
    val_dense = torch.FloatTensor(vr_sparse.toarray()).to(device)
    outputs, _ = model(val_dense)
    mask = val_dense != 0
    mse_loss = criterion(outputs[mask], val_dense[mask])
    rmse = math.sqrt(mse_loss.item())
    print(f"Final Validation Loss (MSE): {mse_loss.item():.4f}")
    print(f"Final Validation RMSE: {rmse:.4f}")

# # Predicting on Test Data
# test_data = pd.read_csv('test.csv')

# # Map IDs to indices
# test_data['user_idx'] = test_data['user_id'].map(user_id_to_index)
# test_data['item_idx'] = test_data['book_id'].map(item_id_to_index)

# # Handle unknown users/items
# test_data = test_data.dropna(subset=['user_idx', 'item_idx'])

# user_indices = test_data['user_idx'].astype(int).values
# item_indices = test_data['item_idx'].astype(int).values

# model.eval()
# with torch.no_grad():
#     full_matrix = torch.zeros(n_m, n_u).to(device)
#     outputs, _ = model(full_matrix)
#     predictions = outputs.cpu().numpy()
#     predicted_ratings = predictions[item_indices, user_indices]

# test_data['predicted_rating'] = predicted_ratings
# test_data.to_csv('test_predictions.csv', index=False)
