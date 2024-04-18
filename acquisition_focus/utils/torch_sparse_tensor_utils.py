import torch



def get_sub_sp_tensor(sp_tensor, eq_value: tuple):
    values = sp_tensor._values()
    indices = sp_tensor._indices()
    sub_marker = torch.tensor([False]*values.numel())

    for val in eq_value:
        sub_marker |= sp_tensor._values() == val

    return torch.sparse_coo_tensor(
        indices[:,sub_marker],
        values[sub_marker], size=sp_tensor.size()
    )



def replace_sp_tensor_values(sp_tensor, existing_values:tuple, new_values:tuple):
    values = sp_tensor._values()
    updated_values = values.clone()
    indices = sp_tensor._indices()

    for c_val, n_val in zip(existing_values, new_values):
        updated_values[values == c_val] = n_val

    return torch.sparse_coo_tensor(
        indices,
        updated_values, size=sp_tensor.size()
    )



def get_inertia_tensor(label):
    # see https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
    assert label.dim() == 3

    if label.is_sparse:
        sp_label = label
    else:
        sp_label = label.to_sparse()
    idxs = sp_label._indices()
    center = idxs.float().mean(1)
    dists = idxs - center.view(3,1)
    r2 = torch.linalg.vector_norm(dists, 2, dim=0)**2
    I = torch.zeros(3,3)

    for i in range(3):
        x_i = dists[i]
        for j in range(3):
            x_j = dists[j]
            kron = float(i==j)
            I[i,j] = (r2 * kron - x_i * x_j).sum()
    # print("inertia\n", I)
    return center, I



def get_center_and_median(label):
    # see https://en.wikipedia.org/wiki/Moment_of_inertia#Inertia_tensor
    assert label.dim() == 3

    if label.is_sparse:
        sp_label = label
    else:
        sp_label = label.to_sparse()

    idxs = sp_label._indices()
    if idxs.numel() == 0:
        none_ret = torch.as_tensor(label.shape).to(label.device) / 2.
        return none_ret, none_ret

    center = idxs.float().mean(1)
    median = idxs.float().median(1).values

    return center, median



def get_main_principal_axes(I):
    assert I.shape == (3,3)
    eigenvectors = torch.linalg.eig(I).eigenvectors.real.T
    eigenvalues = torch.linalg.eig(I).eigenvalues.real
    sorted_vectors = eigenvectors[eigenvalues.argsort()]

    return sorted_vectors[0], sorted_vectors[1], sorted_vectors[2] #min, mid, max