import numpy as np
import h5py
import open3d as o3d
import matplotlib.pyplot as plt
import time


def uniform_2_sphere_fixed(where): #I order the camera not to go under the liver
    #"where" says which lobe has been deformed (left or right). Based on this, I choose a theta that captures the deformation.
    """Generates a random direction on the unit sphere with fixed azimuthal angle (φ) in [0, π]."""
    # Randomly sample theta (azimuthal angle, phi) between 0 and pi (range of 0 to 180 degrees)
    
    if where == "L":
        theta = np.random.uniform(np.radians(240), np.radians(265))  # select your preferred range! (Finding a larger range which is good for all the 20 livers might not be trivial)
    elif where == "R":
        theta = np.random.uniform(np.radians(265), np.radians(290))  # select your preferred range!

    radius = 2 
    phi = np.radians(90) # select your preferred range! (you can do as before if you prefer)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)


    return np.array([x, y, z]), radius

def plot_double(cloud1, cloud2, pov):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot deformed points
    ax.scatter(cloud2[:, 0], cloud2[:, 1], cloud2[:, 2], c='blue', marker='*', s=20, label='Input2')

    # Plot original points
    ax.scatter(cloud1[:, 0], cloud1[:, 1], cloud1[:, 2], c='red', marker='.', s=1, label='Input1')

    # plot POV
    ax.scatter(pov[0], pov[1], pov[2], c='green', marker='x', s=100, label='POV')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()
    plt.show()

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

def plot_single2(cloud1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot deformed points

    # Plot original points
    ax.scatter(cloud1[:, 0], cloud1[:, 1], cloud1[:, 2], c='red', marker='.', s=20, label='Deformed')

    # plot POV
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.legend()
    plt.show()

def plot_single(original_points, pov):
    original_pc = o3d.geometry.PointCloud()
    original_pc.points = o3d.utility.Vector3dVector(original_points)
    original_pc.paint_uniform_color([1, 0, 0])  # Red for original

    pov_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)  # Adjust radius as needed
    pov_sphere.translate(pov)  # Move the sphere to the POV location
    pov_sphere.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([original_pc, pov_sphere])



def find_anchors(points, N, idx): #points closests to these
    
    if idx in (7, 15, 17, 19):  #each liver (or other anatomy) might need its custom P
        P = np.array([0, 0.25, 0.25])
    elif idx == 12:
        P = np.array([0.25, 0.5, 0.25])
    else:
        P = np.array([0.25, 0.25, 0.25])

    dist_Psx = np.linalg.norm(points - P, axis=1) #distanza dal punto normalizzato
    anchors_idx = np.argsort(dist_Psx)[:N]
    anchors_points = points[anchors_idx]

    # I use this to empirically find the best position for P and the optimal number of anchors
    # plot_double(points, anchors_points, P)

    return anchors_points, anchors_idx


#======================== DEF ======================================
def move_lobeR(points, controls_dx_idx):
    disp_dx = np.array([np.random.uniform(-0.25, 0.25), 0.0, np.random.uniform(-0.25, 0.25)])  #choose the displacement you prefer
    # disp_dx = np.array([ 0, 0.0, 0.25])
    final_dx_pos = points[controls_dx_idx]+disp_dx

    return final_dx_pos

def move_lobeL(points, controls_sx_idx):
    disp_sx = np.array([np.random.uniform(-0.25, 0.25), 0.0, np.random.uniform(-0.25, 0.25)])
    final_sx_pos = points[controls_sx_idx]+disp_sx

    return final_sx_pos

def fix_lobeL(points, controls_sx_idx): #I don't apply any displacement, it's like an anchor 
    final_sx_pos = points[controls_sx_idx]

    return final_sx_pos

def fix_lobeR(points, controls_dx_idx): #I don't apply any displacement, it's like an anchor 
    final_dx_pos = points[controls_dx_idx]

    return final_dx_pos
#===========================================================================



def apply_pressure(points, normals, anchors_idx, anchors_points):
    # 1) Build your visibility mask exactly as before
    P_cen = np.array([0, -1.1, 0])  # only used to define 'visibility_mask'
    dist_Pcen = np.linalg.norm(points - P_cen, axis=1)
    dir = np.array([0, -1, 0.2])
    dot_products = normals.dot(dir)
    visibility_mask = dot_products > np.cos(np.radians(40))

    # 2) Define the array of center‐points P₁…Pₙ
    x_coords = np.arange(-0.75, 0.751, 0.25)  # [-0.75, -0.5, ..., +0.75]
    control_indices = []

    for x in x_coords:
        P_i = np.array([x, -1.1, 0.3])
        dists = np.linalg.norm(points - P_i, axis=1)

        # restrict to those that are “visible”
        valid_idx = np.nonzero(visibility_mask)[0]
        valid_dists = dists[valid_idx]

        # if fewer than 5 visible here, take them all; else take 5 closest
        k = min(5, len(valid_idx))
        closest_k = valid_idx[np.argsort(valid_dists)[:k]]
        control_indices.extend(closest_k.tolist())

    # (Optional) remove duplicates so you don’t pin the same point twice
    control_indices = list(dict.fromkeys(control_indices))
    cp = np.array(control_indices, dtype=int)

    # 3) Compute displacements & final positions
    F = np.random.uniform(0.075, 0.125)
    displacements = -normals[cp] * F
    final_pos = points[cp] + displacements

    # 4) Build Open3D constraint objects
    #    fixed anchors go last, so we stack [moving; fixed]
    all_ids = control_indices + list(anchors_idx)
    all_pos = np.vstack([final_pos, anchors_points])

    constraint_ids  = o3d.utility.IntVector(all_ids)
    constraint_pos  = o3d.utility.Vector3dVector(all_pos.tolist())
    deformed_idxs   = np.array(control_indices, dtype=int)

    return constraint_ids, constraint_pos, deformed_idxs

def lobes_deformation(points, anchors_idx, anchors_points):

    N_controls = 10 #20 con 7k
    P_L = np.array([-0.5, -1.1, -0.5]) #left lobe
    P_R = np.array([0.75, -1.1, 0.25]) #right lobe 

    dist_Psx = np.linalg.norm(points - P_L, axis=1) 
    controls_sx_idx = np.argsort(dist_Psx)[:N_controls] #keep the N_controls points closest to P_sx
    dist_Pdx = np.linalg.norm(points - P_R, axis=1)
    controls_dx_idx = np.argsort(dist_Pdx)[:N_controls]

    # I use this to empirically find the best position for P_L or P_R and the optimal number of controls
    # plot_double(points, points[controls_dx_idx], P_R)

    controls_sx_idx_global = controls_sx_idx
    controls_dx_idx_global = controls_dx_idx

    choices = ["L", "R"]
    probabilities = [0.5, 0.5] #random lobe deformation

    # Select one option based on probabilities
    selected_code = np.random.choice(choices, p=probabilities)

    # Execute the corresponding code block
    if selected_code == "L":
        # Left lobe in random horizontal/vertical direction
        final_sx_pos = move_lobeL(points, controls_sx_idx) #====> in this function you can choose the displacement amount
        final_dx_pos = fix_lobeR(points, controls_dx_idx)

    elif selected_code == "R": # Right lobe in random horizontal/vertical direction
        final_dx_pos = move_lobeR(points, controls_dx_idx)
        final_sx_pos = fix_lobeL(points, controls_sx_idx)


    # prepare input to ARAP
    constraint_ids = o3d.utility.IntVector(list(controls_sx_idx_global) + list(controls_dx_idx_global) + list(anchors_idx))
    constraint_pos = o3d.utility.Vector3dVector(np.vstack((final_sx_pos, final_dx_pos, anchors_points)))
    deformed_idxs = np.concatenate((controls_sx_idx_global, controls_dx_idx_global), axis=0)

    return constraint_ids, constraint_pos, deformed_idxs, selected_code


def apply_visibility_mask_crop2(
    distances: np.ndarray,
    visibility_mask: np.ndarray,
    coordinates: np.ndarray,
    N_cropped: int,
    i
) -> np.ndarray:

    # 1) Keep only the "visible" subset
    visible_idx       = np.nonzero(visibility_mask)[0]
    visible_distances = distances[visibility_mask]
    vis_coords        = coordinates[visible_idx]  # shape (M,3)

    if i != 15:
        # 2) Build a mask for points INSIDE that box
        inside_box = (
            (vis_coords[:, 0] >= -0.5) & (vis_coords[:, 0] <=  0.4) &  # x-range
            (vis_coords[:, 1] >   -0.7) &                              # y>−0.7
            (vis_coords[:, 2] >= -0.5) & (vis_coords[:, 2] <=  0.6)    # z-range
        )

        # 3) Exclude those inside the box
        valid_mask       = ~inside_box
        valid_idx        = visible_idx[valid_mask]
        valid_distances  = visible_distances[valid_mask]

        # 4) Sort by distance and pick the top N_cropped
        order            = np.argsort(valid_distances)
        cropped_idx      = valid_idx[order][:N_cropped]
    else:
        order            = np.argsort(visible_distances)
        cropped_idx = visible_idx[order][:N_cropped]

    return cropped_idx


def jitter(pts):
    scale=0.01
    clip=0.05
    noise = np.clip(np.random.normal(0.0, scale, size=(pts.shape[0], 3)),
                    a_min=-clip, a_max=clip)
    pts[:, :3] += noise  # Add noise to xyz

    return pts

def stratified_sample_on_y(
    points: np.ndarray,
    normals: np.ndarray,
    target_n: int = None,
    y_thresh: float = -0.7,
    seed: int = None
):
    """
    Keep all points with y < y_thresh, and randomly sample from the
    others so that the final point-set has exactly target_n points.
    Does *not* shuffle the final list: low-y points come first (in
    original order), then the sampled high-y points in ascending index order.

    Args:
        points (np.ndarray): (N,3) array of XYZ.
        normals (np.ndarray, optional): (N,3) array of normals.
        target_n (int): desired total number of points.
        y_thresh (float): keep all points with y < y_thresh.
        seed (int, optional): for reproducible sampling.

    Returns:
        sampled_points: (target_n,3)
        sampled_normals (if normals given): (target_n,3)
        sampled_idx: indices into the original arrays, in the order they appear.
    """
    if seed is not None:
        np.random.seed(seed)

    N = points.shape[0]
    all_idx = np.arange(N)

    # 1) Split indices by y < threshold vs y >= threshold
    low_y_idx  = all_idx[points[:, 1] < y_thresh]
    high_y_idx = all_idx[points[:, 1] >= y_thresh]

    n_low = len(low_y_idx)

    n_high_needed = target_n - n_low
    if n_high_needed < 0:
        raise ValueError(f"Already have {n_low} points with y<{y_thresh}, "
                        f"which exceeds target {target_n}.")
    if n_high_needed > len(high_y_idx):
        raise ValueError(f"Not enough points with y≥{y_thresh} to sample "
                        f"{n_high_needed} (only have {len(high_y_idx)}).")

    # 2) Randomly choose the needed number from the “high‐y” set
    sampled_high = np.random.choice(high_y_idx, size=n_high_needed, replace=False)

    # 3) Combine low-y and sampled high-y (no shuffle)
    sampled_idx = np.concatenate([low_y_idx, sampled_high])

    return sampled_idx

def plot_corr(cloud1, cloud2, corr):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(cloud1[:, 0], cloud1[:, 1], cloud1[:, 2], 
           c='blue', label='Input1', s=1)
    ax.scatter(cloud2[:, 0], cloud2[:, 1], cloud2[:, 2], 
           c='red', label='Input2', s=1)
    
    for (src_idx, tgt_idx) in corr:
        src_point = cloud1[src_idx]
        tgt_point = cloud2[tgt_idx]
        color = '#54e854'
        
        xs = [src_point[0], tgt_point[0]]
        ys = [src_point[1], tgt_point[1]]
        zs = [src_point[2], tgt_point[2]]
        ax.plot(xs, ys, zs, c=color, linewidth=2)

    ax.legend()
    ax.set_title('Source and Target Clouds with Correspondences')
    plt.show()
    
    
def main():

    src_complete_list = []
    src_complete_def_list = []
    tgt_partial_list = []
    tgt_c2p_masks_list = []
    tgt_complete_list = []
    gt_T_list = []
    corr_complete_list = []
    corr_partial_list = []

    N_dataset = 1 #how many clouds for each liver
    N_livers = 20 #number of livers used in the dataset
    N_samples = N_livers * N_dataset  # total
    N_subsampled = 3500 # number of points for the src
    N_src      = N_subsampled

    crop_ratio = 0.15
    N_cropped = int(N_subsampled * crop_ratio)
    N_partial  = N_cropped
    D          = 3
    N_TRE = 80 #Number of TRE points

    output_path = "dataset_IRCAD.h5" #dataset name
    with h5py.File(output_path, "w") as hf:
    # 1) Create empty, extendable datasets
        ds_src    = hf.create_dataset("src_complete", shape=(N_samples, N_src, D), dtype=np.float32)
        ds_src_def = hf.create_dataset("src_complete_def", shape=(N_samples, N_src, D), dtype=np.float32)
        ds_tgt    = hf.create_dataset("tgt_partial_def", shape=(N_samples, N_partial, D), dtype=np.float32)
        ds_tgt_def = hf.create_dataset("tgt_complete_def", shape=(N_samples, N_src, D), dtype=np.float32)
        ds_overlap = hf.create_dataset("tgt_c2p_mask", shape=(N_samples, N_src), dtype=np.float32)
        ds_gt_T = hf.create_dataset("gt_T", shape=(N_samples, 3, 4), dtype=np.float32)
        ds_corr_complete = hf.create_dataset("corr_complete", shape=(N_samples, N_src, 2), dtype=np.float32)
        ds_corr_partial = hf.create_dataset("corr_partial", shape=(N_samples, N_partial, 2), dtype=np.float32)
        ds_TRE_src = hf.create_dataset("TRE_src", shape=(N_samples, N_TRE, D), dtype=np.float32)
        ds_TRE_tgt = hf.create_dataset("TRE_tgt", shape=(N_samples, N_TRE, D), dtype=np.float32)
        ds_min = hf.create_dataset("min", shape=(N_samples, D), dtype=np.float32)
        ds_max = hf.create_dataset("max", shape=(N_samples, D), dtype=np.float32)

        ij = 0
        perm = np.random.RandomState(3).permutation(N_samples)

        for i in range(1, N_livers+1):

            #Exclude 1 and 2 from trainset --> i used them as testset
            # if (i==1 or i==2):
            #     continue

            print("Liver ", i)
            #read cloud ===> #if necessary: convert to .ply and recompute normals!!!!!!!!!!!!!!!
            ply_path = f"/your/path/to/livers_ply/liver{i}.ply" #path to livers folder
            simplified_mesh = o3d.io.read_triangle_mesh(ply_path)

            vertices = np.asarray(simplified_mesh.vertices)

            # Apply Min-Max normalization on the vertex coordinates
            min_vals = np.min(vertices, axis=0)
            max_vals = np.max(vertices, axis=0)

            # Normalize the coordinates to the [-1, 1] range
            normalized_vertices = 2 * (vertices - min_vals) / (max_vals - min_vals) - 1
            coordinates = normalized_vertices
            normals = np.asarray(simplified_mesh.vertex_normals)

            #I subsample so that every cloud has the same number of points. I only use the indices afterward when I crop. This avoids mesh problems.
            subsampled_idx = stratified_sample_on_y(coordinates, normals=normals, target_n=N_subsampled) #it keeps more points in the visible part and less in the hidden one

            # Assign the normalized coordinates back to the mesh
            simplified_mesh.vertices = o3d.utility.Vector3dVector(normalized_vertices)

            # Find anchors with respect to some fixed point
            num_anchors = 200 
            anchors_points, anchors_idx = find_anchors(coordinates, num_anchors, i)
            # plot_double2(coordinates, coordinates[anchors_idx]) #if you want to visually check the correctness of anchors

            src_complete = coordinates

            random_transformer = RandomTransformSE3(random_mag=False) #rigid transformation

            # Lists to accumulate data for each sample (cycle)

            for idx in range(N_dataset):
                
                #Fixed seed for testset
                # np.random.seed(7+idx)
                
                #Random for training
                np.random.seed()

                print("Generating cloud ", idx)
                
                # start_time = time.time()

                # Simulate a pressure if you want
                press = False #for this work i didnt
                if press:
                    constraint_ids, constraint_pos, deformed_point_idxs = apply_pressure(coordinates, normals, anchors_idx, anchors_points)
                    deformed_mesh = simplified_mesh.deform_as_rigid_as_possible(constraint_ids, constraint_pos, max_iter=50)
                    
                    deformed_points = np.asarray(deformed_mesh.vertices)

                    anchors = np.asarray(simplified_mesh.vertices)[anchors_idx]
                    deformed = np.asarray(simplified_mesh.vertices)[deformed_point_idxs] 
                else:
                    deformed_points = coordinates
                    deformed_mesh = simplified_mesh

                # ========================== NON-RIGID DEFORMATION ==================================
                # apply lobes deformation -> open the function to set the displacement amount and the control points
                # I apply ARAP on the original, normalized point set, NOT on the subsampled!
                constraint_ids, constraint_pos, deformed_point_idxs, where = lobes_deformation(deformed_points, anchors_idx, anchors_points)

                # apply ARAP
                deformed_mesh_fin = deformed_mesh.deform_as_rigid_as_possible(constraint_ids, constraint_pos, max_iter=50)
                
                # Return deformed points
                deformed_points_fin = np.asarray(deformed_mesh_fin.vertices)

                # Use this if you want to check the generated deformation 
                # plot_double2(deformed_points, deformed_points_fin)
                # ==============================================================================


                # ========================== RIGID TRANSFORMATION ==============================
                # Apply a random rigid transformation
                deformed_points_fin_T, gt_T = random_transformer(deformed_points_fin, idx)
                tgt_complete_def = deformed_points_fin_T
                # plot_double2(tgt_complete_def, deformed_points_fin)
                # ==============================================================================


                # ===================================  VISIBLE CROP   ===========================
                #select subsampled index
                normals_sub = normals[subsampled_idx]
                coordinates_sub = coordinates[subsampled_idx]
                tgt_complete_def = deformed_points_fin_T[subsampled_idx]
                
                #find virtual camera POV
                rand_dir, radius = uniform_2_sphere_fixed(where)
                centroid = np.mean(coordinates, axis=0)
                pov = centroid + radius * rand_dir

                max_normal_angle = 80
                dot_products = np.einsum('ij,j->i', normals_sub, rand_dir) #|a|*|b|*cos(theta) = cos(theha), since a and b have norm=1
                visibility_mask = dot_products > np.cos(np.radians(max_normal_angle))

                # Compute distances from the camera
                distances = np.linalg.norm(coordinates_sub - pov, axis=1)  # Distance to camera POV. 

                # I find the points that are visible from that POV. The calculations are done on the ORIGINAL cloud!!
                visible_point_idx = apply_visibility_mask_crop2(distances, visibility_mask, coordinates_sub, N_cropped, i)

                # Create the final mask
                final_mask = np.zeros(len(coordinates_sub), dtype=bool)
                final_mask[visible_point_idx] = True

                # Check if the selected indexes make sense!!
                # plot_double2(tgt_complete_def, tgt_complete_def[final_mask])
                # ==============================================================================


                # ===================================  SHUFFLE and JITTER TGT   ===========================
                src_complete_def = tgt_complete_def
                src_complete, tgt_complete_def, corr_complete, perm_tgt = shuffle_points(coordinates_sub, tgt_complete_def)
                tgt_c2p_mask = final_mask[perm_tgt] #Up until now the mask had aligned indices, now I reorder it with those of the tgt permutation

                cropped_points = tgt_complete_def[tgt_c2p_mask, :] #cropping points on the deformed cloud
                tgt_partial_def = cropped_points

                tgt_partial_def = jitter(tgt_partial_def)

                # Check if you like the result!
                # plot_double2(tgt_partial_def, tgt_complete_def)
                # plot_double2(tgt_partial_def, src_complete)
                # ==============================================================================
                
                # ===================================  FIND GT CORRESPONDENCES   ===============
                corr_partial = update_correspondences(corr_complete, tgt_c2p_mask)
                
                # !!!! check if they make sense !!!!
                # plot_corr(src_complete, tgt_partial_def, corr_partial)
                # ==============================================================================
                
                normalized = True #True -> keep normalized data; False -> return to original dimension
                if not normalized:
                    src_complete = (src_complete + 1) * (max_vals - min_vals) / 2 + min_vals
                    tgt_complete_def = (tgt_complete_def + 1) * (max_vals - min_vals) / 2 + min_vals
                    tgt_partial_def = (tgt_partial_def + 1) * (max_vals - min_vals) / 2 + min_vals
                    src_complete_def = (src_complete_def + 1) * (max_vals - min_vals) / 2 + min_vals

                # TESTSET: find the necessary points for TRE with FPS
                TRE_src, TRE_tgt = sample_unseen_fps(src_complete, src_complete_def, final_mask, N_TRE, seed=None)

                # end_time = time.time()
                # time_passed = end_time - start_time
                # print("time passed ", time_passed)
                
                src_complete_list.append(src_complete)
                src_complete_def_list.append(src_complete_def)
                tgt_complete_list.append(tgt_complete_def)
                tgt_partial_list.append(tgt_partial_def)
                gt_T_list.append(gt_T)
                tgt_c2p_masks_list.append(tgt_c2p_mask)
                corr_complete_list.append(corr_complete)
                corr_partial_list.append(corr_partial)

                print(src_complete)
                print("\n", tgt_partial_def)

                shuffled_idx = perm[ij]
                ds_src[shuffled_idx, ...] = src_complete
                ds_src_def[shuffled_idx, ...] = src_complete_def
                ds_tgt[shuffled_idx, ...] = tgt_partial_def
                ds_tgt_def[shuffled_idx, ...] = tgt_complete_def
                ds_overlap[shuffled_idx, ...] = tgt_c2p_mask
                ds_gt_T[shuffled_idx, ...] = gt_T
                ds_corr_complete[shuffled_idx, ...] = corr_complete
                ds_corr_partial[shuffled_idx, ...] = corr_partial
                ds_TRE_src[shuffled_idx, ...] = TRE_src
                ds_TRE_tgt[shuffled_idx, ...] = TRE_tgt
                ds_min[shuffled_idx, ...] = min_vals
                ds_max[shuffled_idx, ...] = max_vals

                ij += 1

    print(f"Dataset saved to {output_path}")


def farthest_point_sampling(points, N, seed=None):
    """
    Greedy farthest‐point sampling on a point cloud.

    """
    M = points.shape[0]
    rng = np.random.default_rng(seed)
    
    # 1) pick first point at random
    idx0 = rng.integers(0, M)
    idxs = [idx0]
    
    # 2) initialize min_dists to distance from all points to the first
    diff = points - points[idx0:idx0+1]    # (M,3)
    min_dists = np.linalg.norm(diff, axis=1)
    
    # 3) greedily pick farthest points
    for _ in range(1, N):
        next_idx = int(np.argmax(min_dists))
        idxs.append(next_idx)
        # update the min_dists array
        diff = points - points[next_idx:next_idx+1]
        dists = np.linalg.norm(diff, axis=1)
        min_dists = np.minimum(min_dists, dists)
    
    return np.array(idxs)

def sample_unseen_fps(src_complete, src_complete_def, final_mask, N=80, seed=None):
    unseen_idxs = np.where(~final_mask)[0]
    pts_unseen = src_complete_def[unseen_idxs]    
    chosen_local = farthest_point_sampling(pts_unseen, N, seed=seed)
    chosen_idx = unseen_idxs[chosen_local]
    tgt_targets = src_complete_def[chosen_idx]
    src_targets = src_complete[chosen_idx]
    return src_targets, tgt_targets

def shuffle_points(src, tgt):

    ref_permute = np.random.permutation(tgt.shape[0])

    tgt = tgt[ref_permute, :]

    ref_idx_map = np.full(tgt.shape[0], -1)
    ref_idx_map[ref_permute] = np.arange(tgt.shape[0])
    
    src_idx_map = np.full(src.shape[0], -1)
    src_idx_map = np.arange(src.shape[0])

    corr = np.stack([src_idx_map, ref_idx_map], axis=1)

    return src, tgt, corr, ref_permute

def update_correspondences(corr, mask):
    """
    Updates the correspondences after applying a mask to the target point cloud.
    
    Args:
        corr (np.ndarray): An array of shape [N, 2] (or [2, N]) containing correspondences 
                           between src_complete and tgt_complete_def.
                           Each row is [src_idx, tgt_idx].
        mask (np.ndarray): A boolean array of length equal to the number of points in 
                           tgt_complete_def. True for points to keep.
    
    Returns:
        updated_corr (np.ndarray): An array of shape [M, 2] containing the updated correspondences
                                   between src_complete and tgt_partial_def, where M ≤ N.
    """
    # Create a mapping from old target indices to new ones.
    new_indices = -np.ones_like(mask, dtype=int)
    new_idx = 0
    for i, keep in enumerate(mask):
        if keep:
            new_indices[i] = new_idx
            new_idx += 1

    # Update the correspondences: keep only rows where the target index is kept,
    # and update the target index using new_indices.
    updated_corr = []
    for row in corr:
        src_idx, tgt_idx = row
        # If the target point is kept:
        if new_indices[tgt_idx] != -1:
            updated_corr.append([src_idx, new_indices[tgt_idx]])
    
    updated_corr = np.array(updated_corr)
    return updated_corr



class RandomTransformSE3:
    def __init__(self, random_mag: bool = False):
        """Applies a random rigid transformation to the source point cloud

        Args:
            rot_mag (float): Maximum rotation in degrees
            trans_mag (float): Maximum translation T. Random translation will
              be in the range [-X,X] in each axis
            random_mag (bool): If true, will randomize the maximum rotation, i.e. will bias towards small
                               perturbations
        """
        self._rot_mag = 45
        self._trans_mag = 0.5
        self._random_mag = random_mag

    def generate_transform(self):

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        anglex = np.random.uniform() * np.pi * rot_mag / 180.0
        angley = np.random.uniform() * np.pi * rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-trans_mag, trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        return rand_SE3

    def apply_transform(self, p0, transform_mat):
        p1 = self.se3_transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = self.so3_transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        igt = transform_mat


        return p1, igt, igt
    
    def se3_inv(self, pose):
        """Inverts the SE3 transform"""
        rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
        irot = rot.transpose(-1, -2)
        itrans = -irot @ trans
        return self.se3_init(irot, itrans)
    
    def se3_init(self, rot, trans):
        pose = np.concatenate([rot, trans], axis=-1)
        return pose


    def se3_transform(self, pose, xyz):
        """Apply rigid transformation to points

        Args:
            pose: ([B,] 3, 4)
            xyz: ([B,] N, 3)

        Returns:

        """

        assert xyz.shape[-1] == 3 and pose.shape[:-2] == xyz.shape[:-2]

        rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
        transformed = np.einsum('...ij,...bj->...bi', rot, xyz) + trans.transpose(-1, -2)  # Rx + t

        return transformed

    def transform(self, tensor):
        transform_mat = self.generate_transform()
        return self.apply_transform(tensor, transform_mat)

    def __call__(self, tensor, idx):
        transf_cloud, gt_T, transform_s_r = self.transform(tensor)

        return transf_cloud, gt_T 





if __name__ == '__main__':
    main()

