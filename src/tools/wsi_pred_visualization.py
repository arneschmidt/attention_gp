import argparse
import os
import pandas as pd
import numpy as np
from skimage import io
from scipy.interpolate import griddata

parser = argparse.ArgumentParser(description="Cancer Classification")
parser.add_argument("--wsi", "-w", type=int, default=0,
                    help="WSI number")
args = parser.parse_args()
print("WSI number: " + str(args.wsi))

test_df_path = '~/datasets/Panda/Panda_patches_resized_test/test_patches.csv'
preds_path = './dataset_dependent/panda/experiments/initial_experiments/default_preds/instance_preds.csv'
image_path = '~/datasets/Panda/Panda_patches_resized_test/images/'
mask_path = '~/datasets/Panda/Panda_patches_resized_test/masks/'
# test_df_path = '/home/arne/datasets/SICAPv2/partition_mil/Test/Test.xlsx'
# preds_path = './dataset_dependent/sicapv2/gp/split_test/instance_preds.csv'
# image_path = '/home/arne/datasets/SICAPv2/images/'
# mask_path = '/home/arne/datasets/SICAPv2/masks/'
out_path = './out_wsi11/'

#


dataset = 'panda'
res = 512
start_wsi = 0
stop_wsi = 100

if dataset == 'sicap':
    test_df = pd.read_excel(test_df_path)
else:
    test_df = pd.read_csv(test_df_path)

test_df['wsi'] = test_df["image_name"].str.split('_').str[0]
if dataset == 'sicap':
    test_df['x_pos'] = test_df["image_name"].str.split('_').str[4].astype(int)
    test_df['y_pos'] = test_df["image_name"].str.split('_').str[5].astype(int)
else:
    test_df['x_pos'] = test_df["image_name"].str.split('_').str[1].astype(int)
    test_df['y_pos'] = test_df["image_name"].str.split('_').str[2].str.split('.').str[0].astype(int)
preds = pd.read_csv(preds_path)
preds['instances'] = preds['instance_names'].str.split('images/').str[1]
os.makedirs(out_path, exist_ok=True)

wsi_list_df = np.unique(test_df['wsi'])
wsi_list_preds = np.unique(preds['bag_name'])
wsi_list = [np.intersect1d(wsi_list_df, wsi_list_preds)[args.wsi]]
wsi_list = ['1bf0cb1e3f0b2ce35b9b0a406cd225c5']
print(wsi_list)

def blend(image1, image2, alpha=0.5):
    if len(image2.shape) == 2 and len(image1.shape) != 2 :
        ones = np.ones_like(image1)
        image2 = ones * np.reshape(image2, newshape=[image2.shape[0], image2.shape[1], 1])
        # image2 = np.broadcast_to(image2, [image2.shape[0], image2.shape[1], 3])
    blended = alpha * image1 + (1 - alpha) * image2
    return blended

def transform_into_rgb(image, pixel_value=(256, 256, 256)):
    image = np.expand_dims(image, 2)
    image_new = np.concatenate([image*pixel_value[0], image*pixel_value[1], image*pixel_value[2]], axis=2)
    return image_new


for wsi_name in wsi_list:
    print(wsi_name)
    # try:
    wsi_test_df = test_df.loc[test_df['wsi'] == wsi_name]
    wsi_preds = preds.loc[preds['bag_name'] == wsi_name]

    if np.max(wsi_preds['instance_labels']) == 0:
        print('Negative WSI')
        continue


    if len(wsi_test_df) != len(wsi_preds):
        print('Attention! DFs not same length! len(wsi_test_df):' + str(len(wsi_test_df))
              + '; len(wsi_preds):' + str(len(wsi_preds)))

    n_patches_x = np.max(wsi_test_df['x_pos'].astype('int')) + 1
    n_patches_y = np.max(wsi_test_df['y_pos'].astype('int')) + 1
    wsi_x_height = int((n_patches_x * res) / 2 + res)
    wsi_y_width = int((n_patches_y * res) / 2 + res)

    mgrid_pixels = np.mgrid[0:wsi_x_height, 0:wsi_y_width]
    mgrid_preds = np.mgrid[0:n_patches_x, 0:n_patches_y] * (res/2) + (res/2)
    # mgrid_preds = np.reshape(mgrid_preds, newshape=[n_patches_x, n_patches_y, 2])
    mean_grid = np.zeros(shape=[n_patches_x, n_patches_y])
    std_grid = np.zeros(shape=[n_patches_x, n_patches_y])

    wsi = np.ones(shape=[wsi_x_height, wsi_y_width, 3]) * 256
    if dataset == 'sicap':
        gt = np.zeros(shape=[wsi_x_height, wsi_y_width])
        mean = np.zeros(shape=[wsi_x_height, wsi_y_width])
        std = np.zeros(shape=[wsi_x_height, wsi_y_width])
        pred_gt = np.zeros(shape=[wsi_x_height, wsi_y_width])
    else:
        gt = np.zeros(shape=[wsi_x_height, wsi_y_width, 3])
        mean = np.ones(shape=[wsi_x_height, wsi_y_width]) * 256
        std = np.ones(shape=[wsi_x_height, wsi_y_width]) * 256
        pred_gt = np.zeros(shape=[wsi_x_height, wsi_y_width])


    mean_sub = np.min(wsi_preds['mean'])
    std_sub = np.min(wsi_preds['std'])
    mean_factor = 1/(np.max(wsi_preds['mean']) - mean_sub) * 120
    std_factor = 1/(np.max(wsi_preds['std']) - std_sub) * 100
    # mean_sub = 0
    # std_sub = 0
    # mean_factor = 1/(np.max(wsi_preds['mean']) - mean_sub) * 256
    # std_factor = 1/(np.max(wsi_preds['std']) - std_sub) * 256


    for row in range(len(wsi_test_df)):
        pred_row = preds.loc[preds['instances'] == wsi_test_df['image_name'].iloc[row]]
        patch = io.imread(os.path.join(image_path, wsi_test_df['image_name'].iloc[row]))
        patch_gt = io.imread(os.path.join(mask_path, wsi_test_df['image_name'].iloc[row]))
        x_pos = wsi_test_df['x_pos'].iloc[row]
        y_pos = wsi_test_df['y_pos'].iloc[row]

        if x_pos >= n_patches_x or y_pos >= n_patches_y:
            continue

        x_start = int(x_pos*res/2)
        x_stop = int((x_pos*res/2)+res)
        y_start = int(y_pos*res/2)
        y_stop = int((y_pos*res/2)+res)

        wsi[x_start:x_stop, y_start:y_stop] = patch
        gt[x_start:x_stop, y_start:y_stop] = patch_gt
        #
        # mean[x_start:x_stop, y_start:y_stop] = blend(mean[x_start:x_stop, y_start:y_stop],
        #                                              np.ones([res, res]) * (wsi_preds['mean'].iloc[row] * mean_factor - mean_sub))
        # std[x_start:x_stop, y_start:y_stop] = blend(std[x_start:x_stop, y_start:y_stop],
        #                                              np.ones([res, res]) * (wsi_preds['std'].iloc[row] * std_factor - std_sub))


        mean[int(x_start + res/4):int(x_stop - res/4), int(y_start + res/4):int(y_stop - res/4)] = np.ones([int(res/2), int(res/2)]) * ((np.array(pred_row['mean']) - mean_sub) * mean_factor)
        std[int(x_start + res / 4):int(x_stop - res / 4), int(y_start + res / 4):int(y_stop - res / 4)] = np.ones(
            [int(res / 2), int(res / 2)]) * ((np.array(pred_row['std']) - std_sub) * std_factor)
        pred_gt[int(x_start + res / 4):int(x_stop - res / 4), int(y_start + res / 4):int(y_stop - res / 4)] = np.ones(
            [int(res / 2), int(res / 2)]) * ((np.array(pred_row['instance_labels']) > 0) * 256)
        mean_grid[x_pos, y_pos] = (np.array(pred_row['mean']) - mean_sub) * mean_factor
        std_grid[x_pos, y_pos] = ((np.array(pred_row['std']) - std_sub) * std_factor)

    mean_grid = np.reshape(mean_grid, newshape=[(n_patches_x)* (n_patches_y)])
    std_grid = np.reshape(std_grid, newshape=[(n_patches_x)* (n_patches_y)])
    mgrid_preds = np.concatenate([np.expand_dims(mgrid_preds[0], axis=2), np.expand_dims(mgrid_preds[1], axis=2)], axis=2)
    mgrid_preds = np.reshape(mgrid_preds, newshape=[(n_patches_x)* (n_patches_y), 2])
    mean_grid_out = griddata(mgrid_preds, mean_grid, (mgrid_pixels[0], mgrid_pixels[1]), method='linear', fill_value=0.0)
    std_grid_out = griddata(mgrid_preds, std_grid, (mgrid_pixels[0], mgrid_pixels[1]), method='cubic', fill_value=0.0)

    gt = (gt > 100)*50.0
    if dataset == 'sicap':
        gt = np.expand_dims(gt, 2)
        gt = np.concatenate([gt, np.ones_like(gt) * 256, gt], 2)
        # mean = ((mean < 128) * 256)
        # mean = np.expand_dims(mean, 2)
        # mean = np.concatenate([np.ones_like(mean) * 256, mean, mean], 2)
        # # std = ((std < 80) * 256)
        # std = np.expand_dims(std, 2)
        # std = np.concatenate([np.ones_like(std) * 256, std, std], 2)

    gt_new = np.zeros_like(gt)
    gt_new[:,:,0] = gt[:,:,2]
    gt_new[:,:,1] = gt[:,:,0]
    gt_new[:,:,2] = gt[:,:,1]
    # gt_new[:,:,0] = gt[:,:,0]
    # gt_new[:,:,1] = gt[:,:,1]
    # gt_new[:,:,2] = gt[:,:,0]
    gt = gt_new

    # assert np.max(gt) == 256
    # gt[:,:,1:2] = 0
    wsi = np.clip(wsi, a_min=0.0, a_max=256.0)
    wsi_plus_gt = np.clip(wsi + gt, a_min=0.0, a_max=256.0)
    wsi_plus_mean = blend(wsi_plus_gt, mean)
    wsi_plus_std = blend(wsi_plus_gt, std)
    wsi_plus_pred_gt = np.clip(wsi + transform_into_rgb(mean_grid_out, (0, 1, 0)), a_min=0.0, a_max=256.0)

    mean_grid_out = transform_into_rgb(mean_grid_out, (1, 1, 1))
    # stdgrid_out = transform_into_rgb(std_grid_out, (256, 256, 0))
    # mean_std = blend(mean_grid_out, stdgrid_out)
    # wsi_meangrid_out = blend(wsi_plus_gt, mean_grid_out)
    # mean_std_out = blend(mean_grid_out, stdgrid_out)
    # wsi_meangrid_out = np.clip(wsi + mean_grid_out, a_min=0, a_max=256)
    # # mean_std_out = blend(wsi_plus_gt, mean_std)
    # wsi_mean_std = blend(wsi, mean_std_out)
    # wsi_std = blend(wsi_plus_gt, std_grid_out)
    #
    #
    # io.imsave(os.path.join(out_path, wsi_name + '_meangrid.png'), wsi_meangrid_out)
    # io.imsave(os.path.join(out_path, wsi_name + '_wsi_mean_std.png'), wsi_mean_std)
    # io.imsave(os.path.join(out_path, wsi_name + '_mean_std.png'), mean_std)
    # io.imsave(os.path.join(out_path, wsi_name + '_stdgrid.png'), std_grid_out)
    io.imsave(os.path.join(out_path, wsi_name + '.png'), wsi)
    # io.imsave(os.path.join(out_path, wsi_name + '_gt.png'), gt)
    io.imsave(os.path.join(out_path, wsi_name + '_gt.png'), wsi_plus_gt)
    # io.imsave(os.path.join(out_path, wsi_name + '_mean.png'), wsi_plus_mean)
    # io.imsave(os.path.join(out_path, wsi_name + '_std.png'), wsi_plus_std)
    io.imsave(os.path.join(out_path, wsi_name + '_pred_gt.png'), wsi_plus_pred_gt)
    # except:
    #     print('Error, skip WSI')













