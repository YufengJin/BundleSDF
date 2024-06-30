import numpy as np
import torch
import torch
import torch.nn.functional as F
from utils.slam_external import build_rotation
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T

def build_rotation(q):
    norm = torch.sqrt(
        q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
    )
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device="cuda")
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = (
        torch.tensor(
            [
                [2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        .cuda()
        .float()
        .unsqueeze(0)
        .transpose(1, 2)
    )
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
    )
    return cam

def transform_to_frame(model, time_idx, gaussians_grad, camera_grad):
    """
    Function to transform Isotropic or Anisotropic Gaussians from world frame to camera frame.

    Args:
        params: dict of parameters
        time_idx: time index to transform to
        gaussians_grad: enable gradients for Gaussians
        camera_grad: enable gradients for camera pose

    Returns:
        transformed_gaussians: Transformed Gaussians (dict containing means3D & unnorm_rotations)
    """

    params = model["params"]
    # Get Frame Camera Pose
    if camera_grad:
        cam_rot = F.normalize(model["cam_unnorm_rots"][..., time_idx])
        cam_tran = model["cam_trans"][..., time_idx]
    else:
        cam_rot = F.normalize(model["cam_unnorm_rots"][..., time_idx].detach())
        cam_tran = model["cam_trans"][..., time_idx].detach()
    rel_w2c = torch.eye(4).cuda().float()
    rel_w2c[:3, :3] = build_rotation(cam_rot)
    rel_w2c[:3, 3] = cam_tran

    # Check if Gaussians need to be rotated (Isotropic or Anisotropic)
    if params["log_scales"].shape[1] == 1:
        transform_rots = False  # Isotropic Gaussians
    else:
        transform_rots = True  # Anisotropic Gaussians

    # Get Centers and Unnorm Rots of Gaussians in World Frame
    if gaussians_grad:
        pts = params["means3D"]
        unnorm_rots = params["unnorm_rotations"]
    else:
        pts = params["means3D"].detach()
        unnorm_rots = params["unnorm_rotations"].detach()

    transformed_gaussians = {}
    # Transform Centers of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]
    transformed_gaussians["means3D"] = transformed_pts
    # Transform Rots of Gaussians to Camera Frame
    if transform_rots:
        norm_rots = F.normalize(unnorm_rots)
        transformed_rots = quat_mult(cam_rot, norm_rots)
        transformed_gaussians["unnorm_rotations"] = transformed_rots
    else:
        transformed_gaussians["unnorm_rotations"] = unnorm_rots

    return transformed_gaussians, rel_w2c

def transformed_params2depthplussilhouette(params, w2c, transformed_gaussians):
    # Check if Gaussians are Isotropic
    if params["log_scales"].shape[1] == 1:
        log_scales = torch.tile(params["log_scales"], (1, 3))
    else:
        log_scales = params["log_scales"]
    # Initialize Render Variables
    rendervar = {
        "means3D": transformed_gaussians["means3D"],
        "colors_precomp": get_depth_and_silhouette(
            transformed_gaussians["means3D"], w2c
        ),
        "rotations": F.normalize(transformed_gaussians["unnorm_rotations"]),
        "opacities": torch.sigmoid(params["logit_opacities"]),
        "scales": torch.exp(log_scales),
        "means2D": torch.zeros_like(
            params["means3D"], requires_grad=True, device="cuda"
        )
        + 0,
    }
    return rendervar

def transformed_params2rendervar(params, transformed_gaussians):
    # Check if Gaussians are Isotropic
    if params["log_scales"].shape[1] == 1:
        log_scales = torch.tile(params["log_scales"], (1, 3))
    else:
        log_scales = params["log_scales"]
    # Initialize Render Variables
    rendervar = {
        "means3D": transformed_gaussians["means3D"],
        "colors_precomp": params["rgb_colors"],
        "rotations": F.normalize(transformed_gaussians["unnorm_rotations"]),
        "opacities": torch.sigmoid(params["logit_opacities"]),
        "scales": torch.exp(log_scales),
        "means2D": torch.zeros_like(
            params["means3D"], requires_grad=True, device="cuda"
        )
        + 0,
    }
    return rendervar


def get_depth_and_silhouette(pts_3D, w2c):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth of each gaussian center in camera frame
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1)  # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z)  # [num_gaussians, 1]

    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)

    return depth_silhouette


def accumulate_mean2d_gradient(variables):
    variables["means2D_gradient_accum"][variables["seen"]] += torch.norm(
        variables["means2D"].grad[variables["seen"], :2], dim=-1
    )
    variables["denom"][variables["seen"]] += 1
    return variables


def update_params_and_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group["params"][0], None)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group["params"][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group["params"][0]] = stored_state
        params[k] = group["params"][0]
    return params


def cat_params_to_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [g for g in optimizer.param_groups if g["name"] == k][0]
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat(
                (stored_state["exp_avg"], torch.zeros_like(v)), dim=0
            )
            stored_state["exp_avg_sq"] = torch.cat(
                (stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0
            )
            del optimizer.state[group["params"][0]]
            group["params"][0] = torch.nn.Parameter(
                torch.cat((group["params"][0], v), dim=0).requires_grad_(True)
            )
            optimizer.state[group["params"][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(
                torch.cat((group["params"][0], v), dim=0).requires_grad_(True)
            )
            params[k] = group["params"][0]
    return params


def remove_points(to_remove, params, variables, optimizer):
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ["cam_unnorm_rots", "cam_trans"]]
    for k in keys:
        group = [g for g in optimizer.param_groups if g["name"] == k][0]
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group["params"][0]]
            group["params"][0] = torch.nn.Parameter(
                (group["params"][0][to_keep].requires_grad_(True))
            )
            optimizer.state[group["params"][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(
                group["params"][0][to_keep].requires_grad_(True)
            )
            params[k] = group["params"][0]
    variables["means2D_gradient_accum"] = variables["means2D_gradient_accum"][to_keep]
    variables["denom"] = variables["denom"][to_keep]
    variables["max_2D_radius"] = variables["max_2D_radius"][to_keep]
    if "timestep" in variables.keys():
        variables["timestep"] = variables["timestep"][to_keep]
    return params, variables


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def prune_gaussians(params, variables, optimizer, iter, prune_dict):
    if iter <= prune_dict["stop_after"]:
        if (iter >= prune_dict["start_after"]) and (
            iter % prune_dict["prune_every"] == 0
        ):
            if iter == prune_dict["stop_after"]:
                remove_threshold = prune_dict["final_removal_opacity_threshold"]
            else:
                remove_threshold = prune_dict["removal_opacity_threshold"]
            # Remove Gaussians with low opacity
            to_remove = (
                torch.sigmoid(params["logit_opacities"]) < remove_threshold
            ).squeeze()
            # Remove Gaussians that are too big
            if iter >= prune_dict["remove_big_after"]:
                big_points_ws = (
                    torch.exp(params["log_scales"]).max(dim=1).values
                    > 0.1 * variables["scene_radius"]
                )
                to_remove = torch.logical_or(to_remove, big_points_ws)
            # params, variables = remove_points(to_remove, params, variables, optimizer)
            remove_points(to_remove, params, variables, optimizer)
            torch.cuda.empty_cache()

        # Reset Opacities for all Gaussians
        if (
            iter > 0
            and iter % prune_dict["reset_opacities_every"] == 0
            and prune_dict["reset_opacities"]
        ):
            new_params = {
                "logit_opacities": inverse_sigmoid(
                    torch.ones_like(params["logit_opacities"]) * 0.01
                )
            }
            params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def densify(params, variables, optimizer, iter, densify_dict, debug_level=0):
    # TODO modify based on dynaimc 3D gaussian
    if iter <= densify_dict["stop_after"]:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = densify_dict["grad_thresh"]
        if (iter >= densify_dict["start_after"]) and (
            iter % densify_dict["densify_every"] == 0
        ):
            grads = variables["means2D_gradient_accum"] / variables["denom"]
            grads[grads.isnan()] = 0.0
            to_clone = torch.logical_and(
                grads >= grad_thresh,
                (
                    torch.max(torch.exp(params["log_scales"]), dim=1).values
                    <= 0.01 * variables["scene_radius"]
                ),
            )
            new_params = {
                k: v[to_clone]
                for k, v in params.items()
                if k not in ["cam_unnorm_rots", "cam_trans"]
            }
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params["means3D"].shape[0]

            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[: grads.shape[0]] = grads
            to_split = torch.logical_and(
                padded_grad >= grad_thresh,
                torch.max(torch.exp(params["log_scales"]), dim=1).values
                > 0.01 * variables["scene_radius"],
            )
            n = densify_dict["num_to_split_into"]  # number to split into
            new_params = {
                k: v[to_split].repeat(n, 1)
                for k, v in params.items()
                if k not in ["cam_unnorm_rots", "cam_trans"]
            }
            stds = torch.exp(params["log_scales"])[to_split].repeat(n, 3)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(params["unnorm_rotations"][to_split]).repeat(n, 1, 1)
            new_params["means3D"] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params["log_scales"] = torch.log(
                torch.exp(new_params["log_scales"]) / (0.8 * n)
            )
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params["means3D"].shape[0]

            variables["means2D_gradient_accum"] = torch.zeros(num_pts, device="cuda")
            variables["denom"] = torch.zeros(num_pts, device="cuda")
            variables["max_2D_radius"] = torch.zeros(num_pts, device="cuda")
            to_remove = torch.cat(
                (
                    to_split,
                    torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda"),
                )
            )
            params, variables = remove_points(to_remove, params, variables, optimizer)

            if iter == densify_dict["stop_after"]:
                remove_threshold = densify_dict["final_removal_opacity_threshold"]
            else:
                remove_threshold = densify_dict["removal_opacity_threshold"]
            to_remove = (
                torch.sigmoid(params["logit_opacities"]) < remove_threshold
            ).squeeze()
            if iter >= densify_dict["remove_big_after"]:
                big_points_ws = (
                    torch.exp(params["log_scales"]).max(dim=1).values
                    > 0.01 * variables["scene_radius"]
                )
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)

            torch.cuda.empty_cache()

            if debug_level >=2:
                print(
                    f"[DEBUG] Number of 3D Gaussians: {params['means3D'].shape[0]}, {to_clone.sum()} cloned, {to_split.sum()} split and {to_remove.sum()} removed."
                )

        # Reset Opacities for all Gaussians (This is not desired for mapping on only current frame)
        if (
            iter > 0
            and iter % densify_dict["reset_opacities_every"] == 0
            and densify_dict["reset_opacities"]
        ):
            new_params = {
                "logit_opacities": inverse_sigmoid(
                    torch.ones_like(params["logit_opacities"]) * 0.01
                )
            }
            params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def update_learning_rate(optimizer, means3D_scheduler, iteration):
    """Learning rate scheduling per step"""
    for param_group in optimizer.param_groups:
        if param_group["name"] == "means3D":
            lr = means3D_scheduler(iteration)
            param_group["lr"] = lr
            return lr


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper
