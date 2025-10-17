import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def apply_gt_to_source(transf, src):
    """
    Reads 'gt_T' (shape [N,3,4]) and 'tgt_partial_def' (shape [N,M,3])
    from the HDF5 at h5_path, applies each 3×4 transform to the corresponding
    M×3 point cloud, and returns an array of shape [N,M,3]. If output_h5_path
    is given, also writes a new dataset 'tgt_partial_transformed' there.
    """
    
    N, M, _ = src.shape
    # 2) make homogeneous coords (N,M,4)
    ones    = np.ones((N, M, 1), dtype=src.dtype)
    tgt_h   = np.concatenate([src, ones], axis=2)
    
    # 3) apply each [R|t] via matrix‐multiply: (M,4) @ (4,3) -> (M,3)
    out = np.empty_like(src)
    for i in range(N):
        # gt[i] is (3,4); its transpose is (4,3)
        out[i] = tgt_h[i] @ transf[i].T
    
    return out

def apply_gt_to_target(transf: np.ndarray, src: np.ndarray) -> np.ndarray:
    """
    Args:
        transf:  (N, 3, 4) array of ground‐truth [R | t] transforms
                mapping source->target.
        src:     (N, K, 3) array of source point‐clouds.

    Returns:
        (N, K, 3) array of source points transformed by the *inverse*
        of each [R|t], i.e. brought back into the source frame.
    """
    N, K, _ = src.shape
    out = np.empty_like(src)

    # Turn src into homogeneous coords: (N, K, 4)
    ones = np.ones((N, K, 1), dtype=src.dtype)
    src_h = np.concatenate([src, ones], axis=2)

    for i in range(N):
        T = transf[i]           # (3,4)
        R = T[:, :3]            # (3,3)
        t = T[:,  3]            # (3,)

        # Invert: [R | t]^{-1} = [R^T | -R^T t]
        R_inv = R.T                             # (3,3)
        t_inv = -R_inv @ t                      # (3,)

        # Build a 3x4 inverse transform
        T_inv = np.zeros((3, 4), dtype=src.dtype)
        T_inv[:, :3] = R_inv
        T_inv[:,  3] = t_inv

        # Apply: (K,4) @ (4,3) -> (K,3)
        out[i] = src_h[i] @ T_inv.T

    return out

def clean_axis(ax):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    # Remove the axis lines (the cube frame)
    ax.w_xaxis.line.set_lw(0.)
    ax.w_yaxis.line.set_lw(0.)
    ax.w_zaxis.line.set_lw(0.)

def plot_one(cloud, color):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    clean_axis(ax)

    # Plot deformed points
    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=color, marker='.', s=5, label='cloud')
    ax.legend()
    plt.show()

def plot_double2(cloud1, cloud2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    clean_axis(ax)

    # Plot deformed points
    ax.scatter(cloud2[:, 0], cloud2[:, 1], cloud2[:, 2], c='blue', marker='.', s=20, label='Input2')

    # Plot original points
    ax.scatter(cloud1[:, 0], cloud1[:, 1], cloud1[:, 2], c='red', marker='.', s=5, label='Input1')

    # plot POV
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()
    plt.show()


def main():

    # Path to your file
    h5_path = "dataset_IRCAD.h5"
    with h5py.File(h5_path, "r") as f:
        src_ds = f["/src_complete"]       # shape (180, 3988, 3)
        tgt_ds = f["/tgt_partial_def"]  # shape (180, 598,  3)
        gt_T = f["/gt_T"] 

        print(src_ds.shape)

        src_ds = apply_gt_to_source(gt_T, src_ds)
        
        for i in range(10):
            src = src_ds[i]   
            tgt = tgt_ds[i]   
            print(src)

            plot_double2(src, tgt)
        

if __name__ == '__main__':
    main()

