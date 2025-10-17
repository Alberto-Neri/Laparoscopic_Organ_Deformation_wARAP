# Laparoscopic_Organ_Deformation_wARAP
Method used to generate the training and test dataset for our deformed IRCAD liver experiments, presentend in the paper titled: "*Benchmarking Complete-to-Partial Point Cloud Registration Techniques for Laparoscopic Surgery*". 

The idea is to generate deformed point cloud pairs to train complete-to-partial point cloud registration models. 

The final .h5 dataset includes:
- `src_complete`: the original complete point cloud of the organ. **Typically one _input_ of model**. 
- `src_complete_def`: the complete point cloud after the non-rigid deformation
- `tgt_complete_def`: the complete point cloud after the non-rigid deformation and rigid transformation  
- `tgt_partial_def`: representing the intraoperative point cloud, it's tgt_complete_def cropped. **Typically one _input_ of model**.
- `tgt_c2p_mask`: <ins>Overlap scores</ins> for the src. Typically used as **supervision**. 
- `corr_partial`: GT correspondences (shape: [N_dataset, N_tgt, 2]). Typically used as **supervision**.
- `corr_complete`: GT correspondences (shape: [N_dataset, N_src, 2]). Can be used for debug.          
- `TRE_src`: A set of 3D coordinates on src that can be used **to compute the TRE**.                
- `TRE_tgt`: TRE_src corresponding set on **tgt** that can be used **to compute the TRE**.                           
- `gt_T`: The ground truth rigid transformation. Can be used as **supervision** or **to compute rigid rotation and translation errors**.

## Method
  
For the `livers_ply` folder, I used the livers model provided by [IRCAD](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/).
First I converted them to .ply, ***making sure the normals were correct***. This is very important, and if necessary I recalculated them with Meshlab. 
 
Once you have a folder containing the organs .ply you can run `Gen_Def_IRCAD.py`. For each cloud pair the code:
 1. randomly selects one of the 2 lobes of the `src_complete` and applies a random deformation on it.
 2. randomly applies a rigid transformatation to it.
 3. applies a visibile crop to it, obtaining the `tgt_partial_def`. 
 4. shuffles the points in `tgt_partial_def` and applies gaussian noise.
 5. stores all the necessary ground truths.

### Important Parameters to set
- `N_livers`: the number of livers used in the dataset.
- `N_dataset`: how many cloud pairs for each liver.
- `N_subsampled`: number of points for the `src_complete`.
- `crop_ratio`: the % of src_complete points retained to form `tgt_partial_def`.
  - When `crop_ratio` is too low, there might be some **mismatches error**. This happens because the model finds fewer valid points than expected from the dataset. Increasing `max_normal_angle` can help, since a larger angle admits more points by relaxing the normal-angle filtering.
- `ply_path`: your local path to `livers_ply` folder. 
        
           
       
         
