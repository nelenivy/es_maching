import torch
from tqdm import tqdm

def cosine_similarity_matrix(x1, x2):
    x1_norm = x1 / x1.norm(dim=1)[:, None]
    x2_norm = x2 / x2.norm(dim=1)[:, None]
    return torch.mm(x1_norm, x2_norm.transpose(0, 1))


def batch_nearest_neighbors(embs_1, embs_2, batch_size=32000, K=100, return_dists=False):
    n_batches = (embs_1.size(0) + batch_size - 1) // batch_size
    
    embs = [embs_1, embs_2]
    nearest_indices = [[],[]]
    
    if return_dists:
        dists = [[], []]
        
    for mod_ind in (0, 1):
        other_mod = int(not mod_ind)
        
        for batch_idx in tqdm(range(n_batches), leave=True, position=0):
            start_idx = batch_idx*batch_size
            end_idx = min(start_idx+batch_size, embs[mod_ind].size(0))

            left_set_batch = embs[mod_ind][start_idx:end_idx, :]
            print(left_set_batch.shape, embs[other_mod].shape)
            similarity_matrix = cosine_similarity_matrix(left_set_batch, embs[other_mod])
            
            K = min(K, similarity_matrix.shape[1])
            print('similarity_matrix.shape', similarity_matrix.shape)
            top_k = similarity_matrix.topk(k=K, dim=1)
            nearest_indices[mod_ind].append(top_k.indices)
            
            if return_dists:
                dists[mod_ind].append((1.0 - top_k.values) / 2.0)

        #print('before', len(nearest_indices[mod_ind]))      
        #nearest_indices[mod_ind] = [i for i in nearest_indices[mod_ind] 
        #    if i.shape == nearest_indices[mod_ind][0].shape]
        #print('after', len(nearest_indices[mod_ind]))
        nearest_indices[mod_ind] = torch.concat(nearest_indices[mod_ind], axis=0).cpu().numpy()
        
        if return_dists:
            dists[mod_ind] = torch.concat(dists[mod_ind], axis=0).cpu().numpy()
    
    if return_dists:
        return nearest_indices, dists
    else:
        return nearest_indices
            