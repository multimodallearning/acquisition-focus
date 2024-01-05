import torch
from matplotlib import pyplot as plt
from slice_inflate.utils.nifti_utils import nifti_grid_sample



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

def get_main_principal_axes(I):
    assert I.shape == (3,3)
    eigenvectors = torch.linalg.eig(I).eigenvectors.real.T
    eigenvalues = torch.linalg.eig(I).eigenvalues.real
    sorted_vectors = eigenvectors[eigenvalues.argsort()]

    return sorted_vectors[0], sorted_vectors[1], sorted_vectors[2] #min, mid, max

def proj_to_base_vect(vector, base_vector):
    _,D = vector.shape
    assert D == 3
    base_vector = base_vector / torch.linalg.vector_norm(base_vector,2)
    return vector @ base_vector.view(3,1) * base_vector

def get_fact_min_dist(idxs, center, fact, dir):
    end_vect = center + fact * dir
    idxs = idxs.T

    # Use plain idxs for distance
    current_dists = torch.linalg.vector_norm(idxs - end_vect.view(1,3), 2, dim=1)

    # Use projected idxs (torn to dir line) for distance
    # proj_idxs = proj_to_base_vect((idxs.float()-center), dir) + center
    # current_dists = torch.linalg.vector_norm(proj_idxs - end_vect.view(1,3), 2, dim=1)

    current_min_dist = current_dists.min()
    return current_min_dist

def get_extent_vect(start_fact, end_fact, idxs, center, dir):
    MIN_DIST = 1.73/2 # sqrt(3) vox

    while (end_fact - start_fact) > MIN_DIST:
        new_end_fact = end_fact - (end_fact - start_fact)/2.
        new_end_fact_min_dist = get_fact_min_dist(idxs, center, new_end_fact, dir)

        if new_end_fact_min_dist > MIN_DIST:
            end_fact = new_end_fact
        else:
            start_fact += (end_fact - start_fact)/2.

    return (start_fact+end_fact)/2. * dir

def get_min_max_extent_along_axis(label, dir):
    if label.is_sparse:
        sp_label = label
    else:
        sp_label = label.to_sparse()
    idxs = sp_label._indices()
    center = idxs.float().mean(1)
    init_start_fact = 0.
    init_end_fact = torch.linalg.vector_norm(torch.as_tensor(label.shape, dtype=torch.float),2).item()

    p_positive_extend = center + get_extent_vect(init_start_fact, init_end_fact, idxs, center, dir)
    p_negative_extend = center + get_extent_vect(init_start_fact, init_end_fact, idxs, center, -dir)

    return  p_positive_extend, p_negative_extend



def get_affine_from_center_and_plane_vects(px_center, main_plane_vect, plane_vect_two, shape, space='torch', do_return_normal_three=False):
    assert space in ['pixel', 'torch']
    shape = torch.as_tensor(shape)
    # Normalize
    main_plane_vect /=torch.linalg.norm(main_plane_vect,2)
    plane_vect_two /=torch.linalg.norm(plane_vect_two,2)

    normal_three = torch.cross(main_plane_vect, plane_vect_two)
    normal_three /= torch.linalg.norm(normal_three,2)
    plane_vect_two = torch.cross(normal_three, main_plane_vect)

    affine = torch.eye(4)
    affine[:3,:3] = torch.stack([plane_vect_two, main_plane_vect, normal_three], dim=0)
    affine[:3,-1] = px_center

    if space == 'torch':
        affine[:3,:3] = affine[:3,:3].flip(0,1).T
        # pt_affine[:3,:3] = torch.stack([main_plane_vect.flip(0), plane_vect_two.flip(0), normal_three.flip(0)], dim=1).flip(1) # works as well
        affine[:3,-1] = (2.* affine[:3,-1] / shape - 1.).flip(0)

    if do_return_normal_three:
        return affine, normal_three

    return affine
def display_inertia(sp_label, affine=None):

    def plot_inertia(sp_label):
        VECT_LEN = 50
        # Diplay unrotated
        orig_center, orig_I = get_inertia_tensor(sp_label)
        orig_principals = get_main_principal_axes(orig_I)
        orig_label_slice = sp_label.to_dense().sum(-1)
        plt.figure()
        plt.imshow(orig_label_slice.squeeze().T, interpolation='none')
        mpl_center = orig_center[:2]
        plt.scatter(*mpl_center)
        for p, text_label in zip(orig_principals, ['min_inertia','mid_inertia','max_inertia']):
            z = torch.tensor([0.,0.,1.])
            principal_proj = p - p @ z * z
            principal_dir_2D = principal_proj[:2]
            points = torch.stack((mpl_center, mpl_center+principal_dir_2D*VECT_LEN))
            plt.plot(points[:,0], points[:,1], label=text_label)
        plt.legend()
        plt.show()

    plot_inertia(sp_label.to_sparse())

    if affine is not None:
        # Display rotated
        grid = torch.nn.functional.affine_grid(theta=affine[:3].view(1,3,4), size=[1,1]+list(sp_label.shape), align_corners=False)
        tr_label = torch.nn.functional.grid_sample(sp_label.to_dense().view([1,1]+list(sp_label.shape)).float(),
                                                    grid=grid, align_corners=False, mode='nearest')[0,0].int()
        plot_inertia(tr_label)


def display_clinical_views(volume:torch.Tensor, label:torch.Tensor, volume_affine:torch.Tensor, class_dict: dict, num_sa_slices=3, debug=False):
    assert volume.dim() == 3
    assert label.dim() == 3
    assert label.is_sparse

    clinical_view_affines = get_clinical_cardiac_view_affines(label, volume_affine, class_dict, num_sa_slices, debug=debug)
    unrolled_view_affines = {}
    for view_name, affine in clinical_view_affines.items():
        if view_name == 'ALL_SA':
            for a_idx, uaff in enumerate(affine):
                unrolled_name = f'SA-{a_idx}'
                unrolled_view_affines[unrolled_name] = uaff
        else:
            unrolled_view_affines[view_name] = affine

    fov_mm = torch.tensor([200.,200.,1.])
    fov_vox = torch.tensor([100,100,1])

    fig, axs = plt.subplots(len(unrolled_view_affines)//5+1, 5)
    axs = axs.flatten()

    for ax, (view_name, pt_affine) in zip(axs, unrolled_view_affines.items()):

        image_slice, *_ = nifti_grid_sample(volume[None,None], volume_affine[None], None, fov_mm, fov_vox,
            is_label=False, pre_grid_sample_affine=pt_affine[None], pre_grid_sample_hidden_affine=None, dtype=torch.float32
        )
        label_slice, *_ = nifti_grid_sample(label.to_dense()[None,None], volume_affine[None], None, fov_mm, fov_vox,
            is_label=True, pre_grid_sample_affine=pt_affine[None], pre_grid_sample_hidden_affine=None, dtype=torch.float32
        )

        ax.imshow(image_slice[0,0,...,0].T.flip(0), cmap='gray')
        ax.imshow(label_slice[0,0,...,0].T.flip(0), cmap='magma', alpha=.2, interpolation='none')
        ax.set_title(view_name)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()



def get_clinical_cardiac_view_affines(label: torch.Tensor, volume_affine, class_dict: dict, num_sa_slices:int = 3, debug=False):
    assert label.dim() == 3
    assert 'LV' in class_dict
    assert 'RV' in class_dict
    assert 'MYO' in class_dict
    assert 'LA' in class_dict
    assert num_sa_slices % 2 == 1 # We need a center slice

    # nifti_zooms = (volume_affine[:3,:3]**2).sum(0).sqrt() # Currently zooms are not used

    label_shape = list(label.shape)

    if label.is_sparse:
        sp_label = label
    else:
        sp_label = label.to_sparse()

    sp_myolv_label = get_sub_sp_tensor(sp_label, eq_value=(class_dict['MYO'],class_dict['LV']))
    sp_myolvla_label = get_sub_sp_tensor(sp_label, eq_value=(class_dict['MYO'], class_dict['LV'], class_dict['LA']))

    # 0. Extract axial, sagittal, coronal views
    sagittal_vect = torch.tensor([1.,0.,0.])
    coronal_vect = torch.tensor([0.,1.,0.])
    axial_vect = torch.tensor([0.,0.,1.])
    pt_axial_affine = get_affine_from_center_and_plane_vects(torch.as_tensor(label_shape)//2, sagittal_vect, coronal_vect, label_shape)
    pt_coronal_affine = get_affine_from_center_and_plane_vects(torch.as_tensor(label_shape)//2, axial_vect, sagittal_vect, label_shape)
    pt_sagittal_affine = get_affine_from_center_and_plane_vects(torch.as_tensor(label_shape)//2, coronal_vect, axial_vect, label_shape)

    # 1. Extract LV+MYO centerline
    myolv_center, lv_I = get_inertia_tensor(sp_myolv_label)
    lv_min_principal, *_ = get_main_principal_axes(lv_I)

    # 2. Cut normal to cross(axial, LV centerline) -> p2ch
    # display_inertia(sp_myolv_label, pt_p2ch_affine) # debug
    pt_p2ch_affine, ortho_p2ch_vect = get_affine_from_center_and_plane_vects(myolv_center, lv_min_principal, axial_vect,
                                                                                label_shape, do_return_normal_three=True)
    pix_p2ch_affine = get_affine_from_center_and_plane_vects(myolv_center, lv_min_principal, axial_vect,
                                                                                label_shape, space='pixel')

    pt_p4ch_affine, ortho_p4ch_vect = get_affine_from_center_and_plane_vects(myolv_center, lv_min_principal, ortho_p2ch_vect,
                                                                                label_shape, do_return_normal_three=True)
    # 4. Cut normal to cross(p2ch_normal, p4ch_normal) from base to apex
    p1, p2 = get_min_max_extent_along_axis(sp_myolv_label, lv_min_principal)
    delta_p = p2 - p1

    pt_sa_affines = []
    for sa_idx in range(num_sa_slices):
        p_along_sa = p1 + delta_p * sa_idx/(num_sa_slices-1)
        pt_sa_affines.append(get_affine_from_center_and_plane_vects(p_along_sa, ortho_p2ch_vect, ortho_p4ch_vect, label_shape))

    # 5. Get 4CH view
    # Find MYO,LV,RV min and mid-principals in center SA view
    pix_center_sa_affine = get_affine_from_center_and_plane_vects(p1 + .5 * delta_p, ortho_p2ch_vect, ortho_p4ch_vect, label_shape, space='pixel')
    pt_center_sa_affine = get_affine_from_center_and_plane_vects(p1 + .5 * delta_p, ortho_p2ch_vect, ortho_p4ch_vect, label_shape, space='torch')
    myolvrv_class_nums = (class_dict['MYO'], class_dict['LV'], class_dict['RV'])
    sp_myolvrv_label = get_sub_sp_tensor(sp_label, eq_value=myolvrv_class_nums)
    # Get slice
    center_sa_reoriented_slice_lbl, *_ = nifti_grid_sample(sp_myolvrv_label.to_dense()[None,None], volume_affine[None], None,
        fov_mm=torch.tensor([200.,200.,1.]), fov_vox=torch.tensor([200,200,1]),
        is_label=True, pre_grid_sample_affine=pt_center_sa_affine[None], pre_grid_sample_hidden_affine=None, dtype=torch.float32
    )
    if debug:
        display_inertia(center_sa_reoriented_slice_lbl[0,0]) # debug

    # Analyze slice
    _, myo_lv_rv_I = get_inertia_tensor(center_sa_reoriented_slice_lbl[0,0])
    sa_space_myo_lv_rv_min_principal, sa_space_myo_lv_rv_mid_principal, _ = get_main_principal_axes(myo_lv_rv_I) # lives in SA slice space
    # Get MYO,LV,RV min and mid principal direction with respect to volume
    volume_space_sa_myolvrv_mid_principal = pix_center_sa_affine[:3,:3].inverse() @ sa_space_myo_lv_rv_mid_principal
    volume_space_sa_myolvrv_min_principal = pix_center_sa_affine[:3,:3].inverse() @ sa_space_myo_lv_rv_min_principal

    # Extract min-principal of MYO,LV,LA in p2CH slice view
    # Get slice
    p2ch_reoriented_slice_lbl, *_ = nifti_grid_sample(sp_myolvla_label.to_dense()[None,None], volume_affine[None], None,
        fov_mm=torch.tensor([200.,200.,1.]), fov_vox=torch.tensor([200,200,1]),
        is_label=True, pre_grid_sample_affine=pt_p2ch_affine[None], pre_grid_sample_hidden_affine=None, dtype=torch.float32
    )
    if debug:
        display_inertia(p2ch_reoriented_slice_lbl[0,0]) # debug

    # Analyze slice
    _, p2ch_lv_I = get_inertia_tensor(p2ch_reoriented_slice_lbl[0,0])
    p2CH_space_lv_min_principal, *_ = get_main_principal_axes(p2ch_lv_I)
    # Get MYO,LV,LA min-principal direction with respect to volume
    volume_space_p2CH_lv_min_principal = pix_p2ch_affine.inverse()[:3,:3] @ p2CH_space_lv_min_principal

    # Get 4CH affine
    pt_4ch_affine = get_affine_from_center_and_plane_vects(myolv_center, volume_space_sa_myolvrv_min_principal,
                                                           volume_space_p2CH_lv_min_principal, label_shape)
    pix_4ch_affine = get_affine_from_center_and_plane_vects(myolv_center, volume_space_sa_myolvrv_min_principal,
                                                            volume_space_p2CH_lv_min_principal, label_shape, space='pixel')

    # 6. Get 2CH view
    # Find MYO,LV min-principal in 4CH view
    # Get 4CH slice
    _4CH_reoriented_slice_lbl, *_ = nifti_grid_sample(sp_myolvla_label.to_dense()[None,None], volume_affine[None], None,
        fov_mm=torch.tensor([200.,200.,1.]), fov_vox=torch.tensor([200,200,1]),
        is_label=True, pre_grid_sample_affine=pt_4ch_affine[None], pre_grid_sample_hidden_affine=None, dtype=torch.float32
    )
    if debug:
        display_inertia(_4CH_reoriented_slice_lbl[0,0]) # debug
    # Analyze slice
    # TODO: There might be a but when reorienting with nifti-transform and backorienting principal(s) to volume space..
    _, _4CH_myolvla_I = get_inertia_tensor(_4CH_reoriented_slice_lbl[0,0])
    _4CH_space_myolvla_min_principal, *_  = get_main_principal_axes(_4CH_myolvla_I)

    # Get 4CH slice MYO,LV min-principal direction with respect to volume
    myolvla_center, _ = get_inertia_tensor(sp_myolvla_label)
    volume_space_4CH_myolvrv_min_principal = pix_4ch_affine.inverse()[:3,:3] @ _4CH_space_myolvla_min_principal
    pt_2ch_affine = get_affine_from_center_and_plane_vects(myolvla_center, volume_space_sa_myolvrv_mid_principal, volume_space_4CH_myolvrv_min_principal, label_shape)

    return {
        'axial': pt_axial_affine,
        'sagittal': pt_sagittal_affine,
        'coronal': pt_coronal_affine,
        'p2CH': pt_p2ch_affine,
        'p4CH': pt_p4ch_affine,
        'ALL_SA': pt_sa_affines,
        '4CH': pt_4ch_affine,
        '2CH': pt_2ch_affine,
    }
