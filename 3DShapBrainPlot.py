import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio.v2 as imageio  

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import pickle
import os
filename = [
  "A_shap_test_T1.pkl",
  "A_shap_test_T2.pkl",
  "N_shap_test_T1.pkl",
  "N_shap_test_T2.pkl",
  "T_shap_test_T1.pkl",
  "T_shap_test_T2.pkl"
]

for pkl_name in filename:
    with open(os.path.join(r'D:\clinicalAD\ATN\SHAP', pkl_name), 'rb') as f:
        shap_values = pickle.load(f)
    
    print(shap_values.shape)

    fig = plt.figure(figsize=(4,3), dpi=300)  
    ax = fig.add_subplot(111, projection='3d')

    reshaped_data = shap_values 

 
    x, y, z = reshaped_data.nonzero()
    values = reshaped_data[x, y, z]

    sorted_indices = np.argsort(values)

    num_elements = len(values)
    top_percent_index = int(0.005 * num_elements)

    top_values_indices = sorted_indices[-top_percent_index:]
    bottom_values_indices = sorted_indices[:top_percent_index]

    top_x = x[top_values_indices]
    top_y = y[top_values_indices]
    top_z = z[top_values_indices]
    top_values = values[top_values_indices]

    bottom_x = x[bottom_values_indices]
    bottom_y = y[bottom_values_indices]
    bottom_z = z[bottom_values_indices]
    bottom_values = values[bottom_values_indices]

    x = np.concatenate((top_x, bottom_x))
    y = np.concatenate((top_y, bottom_y))
    z = np.concatenate((top_z, bottom_z))
    values = np.concatenate((top_values, bottom_values))

    colors = ['blue', 'white', 'red']  
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)

    norm = mcolors.TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())

    x_scale = 4.0
    y_scale = 1.0
    z_scale = 1.0
    x_scaled = x * x_scale
    y_scaled = y * y_scale
    z_scaled = z * z_scale


    sc = ax.scatter(z_scaled, y_scaled, x_scaled, c=values, cmap=cmap, norm=norm, alpha=0.8, marker='.', s=1)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')


    filenames = []

    for angle in range(0, 360, 5):  
        ax.view_init(elev=30, azim=angle)
        filename = f'frame_{angle}.png'
        plt.savefig(filename, bbox_inches=None, dpi=300)
        filenames.append(filename)

    with imageio.get_writer(pkl_name+'.gif', mode='I', duration=0.1) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in filenames:
        os.remove(filename)
    plt.close() 
