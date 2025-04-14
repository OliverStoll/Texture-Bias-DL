ben_label_mapping = {
    'Coniferous': 'Coniferous Forest',
    'Broad-leaved': 'Broad-leaved F.',
    'Waters': 'Inland Waters',
    'Marine': 'Marine Waters',
    'Arable': 'Arable Land',
    'Wetlands': 'Inland Wetlands',
    'Coastal': 'Coastal Wetlands',
    'Transitional woodland': 'Trans. Woodland',
}
deepglobe_label_mapping = {
    'Urban': 'Urban',
    'Industrial Units': 'Agriculture',
    'Arable': 'Rangeland',
    'Crops': 'Forest',
    'Pastures': 'Water',
    'Cultivation': 'Barren',
}
lower_bounds = {
    'deepglobe':  {'Rangeland': 0.4127, 'Forest': 0.1761, 'Barren': 0.2164, 'Agriculture': 0.6885, 'Water': 0.1811, 'Urban': 0.3197, 'macro': 0.3324, 'micro': 0.4968},
    'bigearthnet':  {'Agriculture': 0.2385, 'Agro-forestry': 0.0274, 'Arable': 0.0025, 'Broad-leaved': 0.1131, 'Coastal': 0.063, 'Coniferous': 0.0334, 'Crops': 0.2348, 'Cultivation': 0.1278, 'Grassland': 0.1082, 'Industrial Units': 0.5251, 'Marine': 0.1375, 'Mixed forest': 0.1319, 'Moors': 0.296, 'Pastures': 0.0027, 'Sand': 0.0057, 'Transitional woodland': 0.0186, 'Urban': 0.1327, 'Waters': 0.2414, 'Wetlands': 0.3481, 'macro': 0.1468, 'micro': 0.2104}
}
upper_bounds = {
    'deepglobe': {'Rangeland': 0.6744, 'Forest': 0.7991, 'Barren': 0.6079, 'Agriculture': 0.9345, 'Water': 0.5881, 'Urban': 0.853, 'macro': 0.7428, 'micro': 0.8215},
    'bigearthnet':   {'Agriculture': 0.8646, 'Agro-forestry': 0.4988, 'Arable': 0.0542, 'Broad-leaved': 0.9028, 'Coastal': 0.625, 'Coniferous': 0.5352, 'Crops': 0.7769, 'Cultivation': 0.446, 'Grassland': 0.9961, 'Industrial Units': 0.9524, 'Marine': 0.8603, 'Mixed forest': 0.481, 'Moors': 0.8426, 'Pastures': 0.307, 'Sand': 0.0441, 'Transitional woodland': 0.4752, 'Urban': 0.8382, 'Waters': 0.7208, 'Wetlands': 0.9493, 'macro': 0.6405, 'micro': 0.8475}
}
transform_to_category = {
    'Bilateral~Channel Shuffle': 'Shape',
    'Bilateral~Patch Shuffle': 'Spectral',
    'Patch Shuffle~Channel Shuffle': 'Texture'
}
transform_names_friendly = {
    'channel_shuffle': 'Channel Shuffle',
    'channel_inversion': 'Channel Inversion',
    'greyscale': 'Channel Averaging',
    'bilateral': 'Bilateral',
    'median': 'Median',
    'gaussian': 'Gaussian',
    'patch_shuffle': 'Patch Shuffle',
    'patch_rotation': 'Patch Rotation',
    'noise': 'Noise',
    'bilateral~patch_shuffle': 'Bilateral & Patch Shuffle',
    'bilateral~channel_shuffle': 'Bilateral & Channel Shuffle',
    'patch_shuffle~channel_shuffle': 'Patch Shuffle & Channel Shuffle',
}
transform_names_data = {
    'channel_shuffle': 'Channel Shuffle',
    'channel_inversion': 'Channel Inversion',
    'greyscale': 'Channel Mean',
    'bilateral': 'Bilateral',
    'median': 'Median',
    'gaussian': 'Gaussian',
    'patch_shuffle': 'Patch Shuffle',
    'patch_rotation': 'Patch Rotation',
    'noise': 'Noise',
    'bilateral~patch_shuffle': 'Bilateral~Patch Shuffle',
    'bilateral~channel_shuffle': 'Bilateral~Channel Shuffle',
    'patch_shuffle~channel_shuffle': 'Patch Shuffle~Channel Shuffle',
}
dataset_names_friendly = {
    'imagenet': 'ImageNet',
    'caltech': 'Caltech',
    'caltech_120': 'Caltech 120',
    'caltech_ft': 'Caltech Finet.',
    'bigearthnet': 'BigEarthNet',
    'rgb_bigearthnet': 'RGB BigEarthNet',
    'deepglobe': 'DeepGlobe',
}
