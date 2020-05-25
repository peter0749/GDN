config = dict(
    full_view = True,
    max_view = 5,
    min_view = 5, # if min_view == max_view: static fusion
    logdir='/tmp3/peter0749/GDN/grasp_logs/pointnet2_direct_loss_fullview_static_5_view_fast',
    representation = 'euler',
    base_loss = 'focal_l1',
    focal_alpha = 0.25,
    focal_gamma = 2.0,
    optimizer = 'adam',
    learning_rate = 0.001,
    grasp_path = '/tmp3/peter0749/GraspDataset/YCB/grasp_labels_npy_nms/good_grasps',
    point_cloud_path = '/tmp3/peter0749/GraspDataset/YCB/all',
    pretrain_path = '/tmp3/peter0749/GDN/grasp_logs/edgeconv_direct_loss_fullview_static_5_view_fast/ckpt/w-31.pth',
    hand_height = 0.11, # 11cm (x range)
    gripper_width = 0.08, # 8cm (y range)
    thickness_side = 0.01, # (z range)
    thickness = 0.01, # gripper_width + thickness*2 = hand_outer_diameter (0.08+0.01*2 = 0.1)
    n_pitch=24,
    n_yaw=25,
    input_dropout_rate=0.2,
    input_points=2048,
    max_grasp_per_object=100,
    max_sample_grasp_per_object=500,
    output_layer=2,
    subsample_levels=[1024, 256, 64],
    epochs=100, # FIXME
    num_workers_dataloader=20,
    num_workers=20,
    batch_size=64,
    sub_division=16,
    eval_split=0.2,
    eval_freq=1,
    eval_pred_threshold=-2e9,
    eval_pred_maxbox=10,
    rot_th=5,
    trans_th=0.02,
    # Tune from multi task module with portion data
    cls_w = 1.19,
    x_w = 6.3,
    y_w = 4.6,
    z_w = 1.93,
    rot_w = 0.19,
    object_list = {
        "train": ["077_rubiks_cube", "065-b_cups", "007_tuna_fish_can", "026_sponge", "011_banana", "021_bleach_cleanser", "016_pear", "014_lemon", "061_foam_brick", "004_sugar_box", "057_racquetball", "065-a_cups", "019_pitcher_base", "048_hammer", "015_peach", "009_gelatin_box", "056_tennis_ball", "071_nine_hole_peg_test", "065-c_cups", "024_bowl", "065-f_cups", "043_phillips_screwdriver", "003_cracker_box", "065-g_cups", "035_power_drill", "055_baseball", "065-d_cups", "029_plate", "033_spatula", "006_mustard_bottle", "052_extra_large_clamp"],
        "val": ["008_pudding_box", "065-e_cups", "037_scissors", "018_plum", "010_potted_meat_can", "017_orange", "005_tomato_soup_can"]
    }
    )

