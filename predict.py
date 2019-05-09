import argparse

import numpy as np
from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import cfg
from label import point_inside_of_quad
from network import East
from preprocess import resize_image
from nms import nms
import pdb
import cv2

def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array, img_path, s):
    geo /= [scale_ratio_w, scale_ratio_h]
    p_min = np.amin(geo, axis=0)
    p_max = np.amax(geo, axis=0)
    min_xy = p_min.astype(int)
    max_xy = p_max.astype(int) + 2
    sub_im_arr = im_array[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
    for m in range(min_xy[1], max_xy[1]):
        for n in range(min_xy[0], max_xy[0]):
            if not point_inside_of_quad(n, m, geo, p_min, p_max):
                sub_im_arr[m - min_xy[1], n - min_xy[0], :] = 255
    sub_im = image.array_to_img(sub_im_arr, scale=False)
    sub_im.save(img_path + '_subim%d.jpg' % s)


def predict(east_detect, img_path, pixel_threshold, quiet=False):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    # pdb.set_trace()
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    with Image.open(img_path) as im:
        im_array = image.img_to_array(im.convert('RGB'))
        d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        quad_im = im.copy()

        out = np.zeros((1, quad_after_nms.shape[1], quad_after_nms.shape[2]))
        for score, geo in zip(quad_scores, quad_after_nms):
            if np.amin(score) > 0:
                out = np.concatenate((out, np.expand_dims(geo, axis=0)),axis=0)
        quad_after_nms = out[1:]

        quad_draw = ImageDraw.Draw(quad_im)
        x_max = quad_after_nms[:,:,0].max()
        x_min = quad_after_nms[:,:,0].min()
        y_max = quad_after_nms[:,:,1].max()
        y_min = quad_after_nms[:,:,1].min()

        # xy0 = (x_max, y_min)
        # xy1 = (x_min, y_min)
        # xy2 = (x_min, y_max)
        # xy3 = (x_max, y_max)
        # quad_draw.line([xy0, xy1, xy2, xy3, xy0], width=2, fill='red')
        # quad_im.save(img_path + '_predict.jpg')
        # exit()

        box = (x_min, y_min, x_max, y_max)
        region = im.crop(box)
        # pdb.set_trace()
        path = img_path.split('.')[0] + '_predict.jpg'
        region.save(path)





# def predict_txt(east_detect, img_path, txt_path, pixel_threshold, quiet=False):
#     img = image.load_img(img_path)
#     d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
#     scale_ratio_w = d_wight / img.width
#     scale_ratio_h = d_height / img.height
#     img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
#     img = image.img_to_array(img)
#     img = preprocess_input(img, mode='tf')
#     x = np.expand_dims(img, axis=0)
#     y = east_detect.predict(x)

#     y = np.squeeze(y, axis=0)
#     y[:, :, :3] = sigmoid(y[:, :, :3])
#     cond = np.greater_equal(y[:, :, 0], pixel_threshold)
#     activation_pixels = np.where(cond)
#     quad_scores, quad_after_nms = nms(y, activation_pixels)

#     txt_items = []
#     for score, geo in zip(quad_scores, quad_after_nms):
#         if np.amin(score) > 0:
#             rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
#             rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
#             txt_item = ','.join(map(str, rescaled_geo_list))
#             txt_items.append(txt_item + '\n')
#         elif not quiet:
#             print('quad invalid with vertex num less then 4.')
#     if cfg.predict_write2txt and len(txt_items) > 0:
#         with open(txt_path, 'w') as f_txt:
#             f_txt.writelines(txt_items)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='demo/002.png',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    img_path = args.path
    threshold = float(args.threshold)
    print(img_path, threshold)
    saved_model_weights_file_path = '/opt/intern/users/jeashen/code/AdvancedEAST/saved_model/east_model_weights_3T736_origin.h5'

    east = East()
    east_detect = east.east_network()
    # east_detect.load_weights(cfg.saved_model_weights_file_path)
    east_detect.load_weights(saved_model_weights_file_path)
    predict(east_detect, img_path, threshold)
