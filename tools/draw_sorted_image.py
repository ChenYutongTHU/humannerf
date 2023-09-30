import os,pickle
image_dir = 'dataset/zju_mocap/387_tava/images'

'''
txtfile = 'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/train_render-metrics.loss-ascending.txt'
outputfile = 'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/train_render-metrics.loss-ascending/{}.png'
lines = open(txtfile).readlines()
image_paths = [os.path.join(image_dir, f'{l.strip()}.png') for l in lines] 
'''

pklfile = 'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest/distance_mat/distance_mat_0.30-0.00.cluster.pkl'
results = pickle.load(open('experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest/distance_mat/distance_mat_0.30-0.00.cluster.pkl','rb'))
for cid, cluster in enumerate(results):
    image_paths = [os.path.join(image_dir, f'{n}')for n in cluster['names']]
    outputfile = f'experiments/human_nerf/zju_mocap/p387/tava/387_l1.0m0.2_4view_last2/latest/distance_mat/distance_mat_0.30-0.00.cluster.pkl_cluster{cid}'+'/{}.png'

    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    from PIL import Image
    import numpy as np
    import glob
    from tqdm import tqdm

    # Image size (assumes all images have the same size)
    image = Image.open(image_paths[0])
    new_size = (256, 256)



    # Create image groups
    image_groups = [image_paths[i:i + 64] for i in range(0, len(image_paths), 64)]

    # Loop through each group
    for i, image_group in enumerate(image_groups):
        # Create an "empty" image to paste the other images into
        combined_image = Image.new('RGB', (new_size[0] * 8, new_size[1] * 8))

        # Loop through each image and paste it into the combined image at the correct position
        for index, image_path in tqdm(enumerate(image_group)):
            image = Image.open(image_path)
            image = image.resize(new_size, Image.ANTIALIAS)  # Resize image
            x = index % 8 * new_size[0]
            y = index // 8 * new_size[1]
            combined_image.paste(image, (x, y))

        # Save the combined image
        # Save the combined image
        combined_image.save(outputfile.format(i))
        print(outputfile.format(i))


    print("Done!")