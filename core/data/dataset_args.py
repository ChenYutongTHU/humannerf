from configs import cfg

class DatasetArgs(object):
    dataset_attrs = {}

    subjects = ['313', '315', '377', '386', '387', '390', '392', '393', '394','xiao']

    if cfg.category == 'human_nerf' and cfg.task == 'zju_mocap':
        for sub in subjects:
            dataset_attrs.update({
                f"pjlab_{sub}_view14_after-800_step4":{ #for novel pose
                    "dataset_path": f"dataset/pjlab/{sub}/view14_after-800_step4",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "subject": sub,                    
                },
                f"pjlab_{sub}_view14_500-800_step5":{ #for novel view
                    "dataset_path": f"dataset/pjlab/{sub}/view14_500-800_step5",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "subject": sub,                    
                },
                f"pjlab_{sub}_view0235_500-800":{
                    "dataset_path": f"dataset/pjlab/{sub}/view0235_500-800",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "subject": sub,                    
                },
                f"pjlab_{sub}_train-all_view00":{
                    "dataset_path": f"dataset/pjlab/{sub}/00",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "subject": sub,                    
                },
                f"zju_{sub}_test_fr-tv_vw-novel-all":{
                    "source_path": f"data/zju/CoreView_{sub}",
                    "dataset_path": f"dataset/zju_mocap/{sub}_fr-tv_vw-novel-all",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,                    
                },    
                f"zju_{sub}_test_fr-tv_vw-3-9-15-22_ood":{
                    "source_path": f"data/zju/CoreView_{sub}",
                    "dataset_path": f"dataset/zju_mocap/{sub}_fr-tv_vw-3-9-15-22_ood",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,                    
                },                 
                f"zju_{sub}_test_fr-hn_vw-3-9-15-22":{
                    "source_path": f"data/zju/CoreView_{sub}",
                    "dataset_path": f"dataset/zju_mocap/{sub}_fr-hn_vw-3-9-15-22",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,                    
                },    
                f"zju_{sub}_test_fr-tv_vw-3-9-15-22":{
                    "source_path": f"data/zju/CoreView_{sub}",
                    "dataset_path": f"dataset/zju_mocap/{sub}_fr-tv_vw-3-9-15-22",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,                    
                },    
                f"zju_{sub}_train": {
                    "dataset_path": f"dataset/zju_mocap/{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "subject": sub,
                },
                f"zju_{sub}_test": {
                    "dataset_path": f"dataset/zju_mocap/{sub}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                },
                f"zju_{sub}_tava_train": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_tava",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "subject": sub,
                },
                f"zju_{sub}_nb_4view_train": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_nb_4view_train",
                    "source_path": f"data/zju/CoreView_{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "subject": sub,
                },
                f"zju_{sub}_nb_1view_test": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_nb_4view_train",
                    "source_path": f"data/zju/CoreView_{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                    "select_views":[1],
                    "skip":4,
                },
                f"zju_{sub}_nb_4view_novelpose": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_nb_4view_novelpose",
                    "source_path": f"data/zju/CoreView_{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                },
                f"zju_{sub}_nb_1view_novelpose": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_nb_1view_novelpose_all",
                    "source_path": f"data/zju/CoreView_{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                },
                f"zju_{sub}_nb_4view_novelview": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_nb_4view_novelview",
                    "source_path": f"data/zju/CoreView_{sub}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                },
                f"zju_387_tava_pose1-529":{
                    "dataset_path": f"dataset/zju_mocap/387_tava_pose1-529",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "subject": sub,                        
                },
                f"zju_387_nb_rightlimb_32":{
                    "dataset_path": f"dataset/zju_mocap/387_nb_pose_rightlimb_32",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "subject": sub,                        
                },
                f"zju_{sub}_tava_train_render": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_tava",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "subject": sub,
                },
                f"zju_{sub}_tava_test": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_tava",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                },
                f"zju_{sub}_tava_train_1view": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_tava_1view",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "subject": sub,
                },
                f"zju_{sub}_tava_test_1view": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_tava_1view",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                }, 
                f"zju_{sub}_tava_train_1view_camera6": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_tava_1view_camera6",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "subject": sub,
                },
                f"zju_{sub}_tava_test_1view_camera6": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_tava_1view_camera6",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                }, 
                f"zju_{sub}_tava_train_1view_camera12": {
                    "source_path": f"data/zju/CoreView_{sub}",
                    "dataset_path": f"dataset/zju_mocap/{sub}_tava_1view_camera12",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "subject": sub,
                },
                f"zju_{sub}_tava_test_1view_camera12": {
                    "source_path": f"data/zju/CoreView_{sub}",
                    "dataset_path": f"dataset/zju_mocap/{sub}_tava_1view_camera12",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                }, 
                f"zju_{sub}_tava_train_1view_camera18": {
                    "source_path": f"data/zju/CoreView_{sub}",
                    "dataset_path": f"dataset/zju_mocap/{sub}_tava_1view_camera18",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "subject": sub,
                },
                f"zju_{sub}_tava_test_1view_camera18": {
                    "source_path": f"data/zju/CoreView_{sub}",
                    "dataset_path": f"dataset/zju_mocap/{sub}_tava_1view_camera18",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                }, 
                f"zju_{sub}_tava_train_2view": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_tava_2view",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "subject": sub,
                },
                f"zju_{sub}_tava_test_2view": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_tava_2view",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                },     
            })


    if cfg.category == 'human_nerf' and cfg.task == 'wild':
        dataset_attrs.update({
            "monocular_train": {
                "dataset_path": 'dataset/wild/monocular',
                "keyfilter": cfg.train_keyfilter,
                "ray_shoot_mode": cfg.train.ray_shoot_mode,
            },
            "monocular_test": {
                "dataset_path": 'dataset/wild/monocular',  
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": 'image',
                "src_type": 'wild'
            },
        })


    @staticmethod
    def get(name):
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()
