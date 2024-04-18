import torch
import torch.cuda.amp as amp



def get_random_affine(rotation_strength=0.2, zoom_strength=0.2, offset_strength=0.0):
    rand_z = torch.rand(1) * zoom_strength - zoom_strength/2 + 1.0

    ortho_vect = torch.tensor((rotation_strength*torch.randn(2)).tolist()+[1.])
    ortho_vect /= ortho_vect.norm(2)
    one = torch.tensor([1.]+(rotation_strength*torch.randn(2)).tolist())
    two = torch.cross(ortho_vect, one)
    two /= two.norm(2)
    one = torch.cross(two, ortho_vect)

    rand_theta_r = torch.eye(4)
    rand_theta_r[:3,:3] = torch.stack([one,two,ortho_vect])
    rand_theta_z = torch.diag(torch.tensor([rand_z,rand_z,rand_z,1.0]))

    rand_theta_t = torch.eye(4)
    rand_theta_t[:3,3] = offset_strength*torch.randn(3)

    return rand_theta_z @ rand_theta_r @ rand_theta_t



def compute_rotation_matrix_from_ortho6d(ortho):
    # see https://github.com/papagina/RotationContinuity/blob/master/Inverse_Kinematics/code/tools.py
    x_raw = ortho[:, 0:3]
    y_raw = ortho[:, 3:6]

    x = x_raw / x_raw.norm(dim=1, keepdim=True)
    z = x.cross(y_raw)
    z = z / z.norm(dim=1, keepdim=True)
    y = z.cross(x)

    # torch.stack([x, y, z], dim=-1)
    r00 = x[:,0]
    r01 = y[:,0]
    r02 = z[:,0]
    r10 = x[:,1]
    r11 = y[:,1]
    r12 = z[:,1]
    r20 = x[:,2]
    r21 = y[:,2]
    r22 = z[:,2]
    zer = torch.zeros_like(r00)
    one = torch.ones_like(r00)

    theta_r = torch.stack(
        [r00, r01, r02, zer,
         r10, r11, r12, zer,
         r20, r21, r22, zer,
         zer, zer, zer, one], dim=1)

    theta_r = theta_r.view(-1,4,4)

    return theta_r



def normal_to_rotation_matrix(normals):
    """Convert 3d vector (unnormalized normals) to 4x4 rotation matrix

    Args:
        normal (Tensor): tensor of 3d vector of normals.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """

    nzs, nys, nxs = normals[:,0], normals[:,1], normals[:,2]

    # see https://math.stackexchange.com/questions/1956699/getting-a-transformation-matrix-from-a-normal-vector
    r00 = nys / torch.sqrt(nxs**2 + nys**2)
    r01 = -nxs / torch.sqrt(nxs**2 + nys**2)
    r02 = torch.zeros_like(nxs)
    r10 = nxs * nzs / torch.sqrt(nxs**2 + nys**2)
    r11 = nys * nzs / torch.sqrt(nxs**2 + nys**2)
    r12 = -torch.sqrt(nxs**2 + nys**2)
    r20 = nxs
    r21 = nys
    r22 = nzs
    zer = torch.zeros_like(nxs)
    one = torch.ones_like(nxs)

    theta_r = torch.stack(
        [r00, r01, r02, zer,
         r10, r11, r12, zer,
         r20, r21, r22, zer,
         zer, zer, zer, one], dim=1)

    theta_r = theta_r.view(-1,4,4)

    return theta_r



def angle_axis_to_rotation_matrix(angle_axis, eps=1e-6):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """
    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2+eps)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    with amp.autocast(enabled=False):
        _angle_axis = _angle_axis.to(torch.float32)
        theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2, eps)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    B = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(B, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4