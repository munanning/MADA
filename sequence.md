A-stage 1
1-save_feat_source.py: get the './features/full_dataset_objective_vectors.pkl'
2-cluster_anchors_source.py: cluster the './features/full_dataset_objective_vectors.pkl' to './anchors/cluster_centroids_full_10.pkl'
3-select_active_samples.py: select active samples with './anchors/cluster_centroids_full_10.pkl' to 'stage1_cac_list_0.05.txt'
4-train_active_stage1.py: train stage1 model with anchors './anchors/cluster_centroids_full_10.pkl' and active samples 'stage1_cac_list_0.05.txt', get the 'from_gta5_to_cityscapes_on_deeplab101_best_model_stage1.pkl', which is stored in the runs/active_from_gta_to_city_stage1

B-stage 2
1-save_feat_target.py: get the './features/target_full_dataset_objective_vectors.pkl.pkl'
2-cluster_anchors_target.py: cluster the './features/target_full_dataset_objective_vectors.pkl' to './anchors/cluster_centroids_full_target_10.pkl'
3-train_active_stage2.py: train stage2 model with anchors './anchors/cluster_centroids_full_target_10.pkl' and active samples 'stage1_cac_list_0.05.txt', get the 'from_gta5_to_cityscapes_on_deeplab101_best_model_stage2.pkl'
