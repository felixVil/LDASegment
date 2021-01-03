from skimage import measure
from skimage.io import imsave
import os
from skimage.draw import polygon, polygon_perimeter
from skimage.io import imread
from skimage.transform import resize
from skimage.morphology import label
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


def fix_array(rect):
    rect = np.round(rect)
    rect = rect.astype(int)
    return rect


def fix_dimensions(im_array):
    im_shape = im_array.shape
    if len(im_shape) > 3 and im_shape[0] == 1:
        fixed_im = np.squeeze(im_array, axis=0)
    elif len(im_shape) == 2:
        fixed_im = np.zeros((im_shape[0], im_shape[1], 1))
        fixed_im[:, :, 0] = np.copy(im_array)
    else:
        fixed_im = np.copy(im_array)
    return fixed_im


def read_image(im_file_path):
    img = imread(im_file_path)
    return img


def create_image_overlaid_with_polygon(img, poly_array, outline_color = 'green', text_overlay = None):
    perimeter_row_coords, perimeter_column_coords = polygon_perimeter(poly_array[1::2], poly_array[0::2])
    img_pil = Image.fromarray(img, 'RGB')
    draw = ImageDraw.Draw(img_pil)
    zipped_list_of_coords = list(zip(perimeter_column_coords, perimeter_row_coords))
    draw.polygon(zipped_list_of_coords, fill=None, outline=outline_color)
    if text_overlay is not None:
        fnt = ImageFont.truetype("C:/Windows/Fonts/Arial/ariblk.ttf", 20)
        draw.text((0, 0), text_overlay, (255, 255, 0), font=fnt)
    img_poly_overlaid = np.array(img_pil.getdata()).reshape(img_pil.size[1], img_pil.size[0], 3)
    img_poly_overlaid = img_poly_overlaid.astype('uint8')
    return img_poly_overlaid


def create_bounding_rect_from_mask(mask_img):
    if mask_img.shape[-1] == 1:
        mask_2d = np.squeeze(mask_img, axis=-1)
    else:
        mask_2d = mask_img
    y_inds, x_inds = np.nonzero(mask_2d > 0.5)
    if y_inds.size != 0 and x_inds.size != 0:
        top_coord = min(y_inds)
        left_coord = min(x_inds)
        bottom_coord = max(y_inds)
        right_coord = max(x_inds)
    else:
        top_coord, left_coord, bottom_coord, right_coord = (np.empty([1]) for _ in range(4))
    return top_coord, left_coord, bottom_coord, right_coord


def create_image_overlaid_with_rotated_rect(img, mask_img):
    _, rotated_bb_4_drawing, _ = convert_mask_to_rotated_rect(mask_img)
    img_overlay = cv2.drawContours(img, [rotated_bb_4_drawing], 0, (0, 255, 0), 3)
    return img_overlay


def create_image_overlaid_with_rect(img, mask_img):
    top_coord, left_coord, bottom_coord, right_coord = create_bounding_rect_from_mask(mask_img)
    img_pil = Image.fromarray(img, 'RGB')
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([left_coord ,top_coord, right_coord, bottom_coord], fill=None, outline=(0, 255, 0, 128), width=3)
    img_poly_overlaid = np.array(img_pil.getdata()).reshape(img_pil.size[1], img_pil.size[0], 3)
    img_poly_overlaid = img_poly_overlaid.astype('uint8')
    return img_poly_overlaid


def create_image_overlaid_with_mask(img, mask_img, mask_color_ind=1):
    alpha = 0.6
    indices = np.nonzero(mask_img)
    img_mask_overlaid = img
    img_mask_overlaid[indices[0], indices[1], mask_color_ind] = img_mask_overlaid[indices[0], indices[1], mask_color_ind] * (1 + alpha)
    return img_mask_overlaid


def convert_rect_to_real_poly(rect_array):
    x = rect_array[0]
    y = rect_array[1]
    width = rect_array[2]
    height = rect_array[3]
    poly_array = np.array([x, y, x + width, y, x + width, y + height, x, y + height])
    return poly_array


def create_mask_from_poly(im_rows, im_cols, poly_array):
    # sometimes polygon is of the form [x, y, width ,height].
    if len(poly_array) == 4:
        poly_array = convert_rect_to_real_poly(poly_array)
    # crop_rect and rect_array(left, upper, right, lower)
    poly_array = poly_array - 1 #convert from matlab coords.
    mask_image = np.zeros((im_rows, im_cols, 1))
    row_coords, column_coords = polygon(poly_array[1::2], poly_array[::2], shape=(im_rows, im_cols))
    row_coords = np.clip(row_coords, 0, im_rows - 1)
    column_coords = np.clip(column_coords, 0, im_cols - 1)
    mask_image[row_coords, column_coords, 0] = 1
    return mask_image


def convert_mask_to_rotated_rect(mask_2d):
    mask_2d = cv2.convertScaleAbs(np.squeeze(mask_2d))
    major = cv2.__version__.split('.')[0]
    if major == '3':
        _, contours, _ = cv2.findContours(mask_2d.copy(), cv2.RETR_EXTERNAL, 1)  # not copying here will throw an error
    else:
        contours, _ = cv2.findContours(mask_2d.copy(), cv2.RETR_EXTERNAL, 1)
    contours = np.concatenate(contours)

    rect = cv2.minAreaRect(contours)  # basically you can feed this rect into your classifier
    #(x, y), (w, h), a = rect  # a - angle
    box = cv2.boxPoints(rect)

    box_4_drawing = np.int0(box)  # turn into ints

    box_poly_format = [box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1],  box[3, 0], box[3, 1]]
    return box_poly_format, box_4_drawing, rect


def create_initial_mask(img, poly_array):
    im_height, im_width, channels = img.shape
    initial_mask = create_mask_from_poly(im_height, im_width, poly_array)
    return initial_mask


def binarize_mask(mask_img, threshold=0.5):
    sel_thresh = min(threshold, mask_img.max())
    mask_2d = np.zeros(mask_img.shape, dtype=int)
    mask_2d[mask_img >= sel_thresh] = 1
    return mask_2d


def get_regions_from_mask(mask_img, threshold=0.5):
    mask_2d = binarize_mask(mask_img, threshold=threshold)
    label_mask = label(mask_2d)
    regions = measure.regionprops(label_mask)
    return regions


def get_single_component_from_mask(mask_img, threshold=0.5):
    if mask_img.shape[-1] == 1:
        mask_img = np.squeeze(mask_img, axis=-1)
    regions = get_regions_from_mask(mask_img, threshold=threshold)
    blob_areas, coords_lists, blob_areas_all, coords_lists_all = (list() for _ in range(4))
    for region in regions:
        blob_areas_all.append(region.area)
        coords_lists_all.append(region.coords)
        blob_areas.append(region.area)
        coords_lists.append(region.coords)

    if not blob_areas:#case when empty blobs
        blob_areas = blob_areas_all
        coords_lists = coords_lists_all

    blob_areas = np.array(blob_areas)
    indx = np.argmax(blob_areas)
    if indx.size > 1:
        indx = indx[0]
    mask_center_component = np.zeros(mask_img.shape, dtype='float64')
    mask_center_component[coords_lists[indx][:, 0], coords_lists[indx][:, 1]] = 1
    mask_center_component = fix_dimensions(mask_center_component)
    return mask_center_component, coords_lists[indx]


def embed_mask_into_image(mask_2d, embedding_rect, im_shape):
    x_rect_size, y_rect_size = get_crop_rect_sizes(embedding_rect)
    resized_back_mask_2d = resize(mask_2d, (y_rect_size, x_rect_size), mode='reflect', preserve_range=True)
    display_mask_standard = np.zeros((im_shape[0], im_shape[1]))
    embedding_rect = embedding_rect.astype(int)
    display_mask_standard[embedding_rect[1]:embedding_rect[3],
    embedding_rect[0]:embedding_rect[2]] = resized_back_mask_2d[:, :]
    return display_mask_standard


def get_crop_rect_sizes(crop_rect):
    y_rect_size = crop_rect[3] - crop_rect[1]
    x_rect_size = crop_rect[2] - crop_rect[0]
    return  x_rect_size, y_rect_size


def save_overlaid_and_mask_images(indx, img_read, predict_mask, result_path):
    img_overlay = create_image_overlaid_with_rotated_rect(img_read, predict_mask)
    img_overlay = create_image_overlaid_with_mask(img_overlay, predict_mask)
    result_img_filename = "res_{:05d}.png".format(indx + 1)
    result_mask_img_filename = "mask_{:05d}.png".format(indx + 1)
    imsave(os.path.join(result_path, result_img_filename), img_overlay)
    predict_mask_tiled = np.tile(predict_mask, (1, 1, 3))
    predict_mask_tiled = predict_mask_tiled * 255
    predict_mask_tiled = predict_mask_tiled.astype('uint8')
    imsave(os.path.join(result_path, result_mask_img_filename), predict_mask_tiled)


def save_mask_image(predict_mask, result_path, result_mask_img_filename):
    predict_mask_tiled = np.tile(predict_mask, (1, 1, 3))
    predict_mask_tiled = predict_mask_tiled * 255
    predict_mask_tiled = predict_mask_tiled.astype('uint8')
    imsave(os.path.join(result_path, result_mask_img_filename), predict_mask_tiled)


def get_region_dict_from_mask(mask_img):
    top_coord, left_coord, bottom_coord, right_coord = create_bounding_rect_from_mask(mask_img)
    region = {'x': int(left_coord), 'y': int(top_coord), 'width': int(right_coord - left_coord), 'height': int(bottom_coord - top_coord),
              'confidence': 1}
    return region


def create_single_crop_rect(alpha, mask, im_width, im_height, input_shape):
    top_coord, left_coord, bottom_coord, right_coord = create_bounding_rect_from_mask(mask)
    major_dimension_x = (right_coord - left_coord) * alpha
    major_dimension_y = (bottom_coord - top_coord) * alpha
    crop_left = round(left_coord - major_dimension_x)
    crop_top = round(top_coord - major_dimension_y)
    crop_right = round(right_coord + major_dimension_x)
    crop_lower = round(bottom_coord + major_dimension_y)
    crop_top = max(0, crop_top)
    crop_left = max(0, crop_left)
    crop_lower = min(im_height, crop_lower)
    crop_right = min(im_width, crop_right)
    crop_rect = np.array([crop_left, crop_top, crop_right, crop_lower])
    return crop_rect


def create_cropped_image(im, crop_rect):
    img = np.copy(im)
    rounded_crop_rect = fix_array(crop_rect)
    im_cropped = img[rounded_crop_rect[1]:rounded_crop_rect[3], rounded_crop_rect[0]:rounded_crop_rect[2]]
    return im_cropped


def norm_mask_by_max(mask):
    mask_max = np.max(mask)
    mask = mask / mask_max
    return mask


def get_mask_center(binary_mask_image):
    foreground_mask_inds = np.nonzero(binary_mask_image)
    y_coords = foreground_mask_inds[0]
    x_coords = foreground_mask_inds[1]
    center = np.mean(np.column_stack((y_coords, x_coords)), axis=0)
    return center


def check_f1(selection_mask, final_map, is_failure=False):
    if is_failure:
        component_mask, _ = get_single_component_from_mask(final_map)
    else:
        component_mask = binarize_mask(final_map)
    if len(component_mask.shape) > 2:
        component_mask = np.squeeze(component_mask, axis=-1)
    f_1 = 2 * np.sum(component_mask * selection_mask[:, :, 0])/np.sum(component_mask + selection_mask[:, :, 0])
    return f_1


def check_if_small_mask(big_mask, threshold_ratio):
    ratio = np.count_nonzero(big_mask) / (big_mask.shape[0] * big_mask.shape[1])
    return ratio < threshold_ratio


def create_tight_rect_poly_array(poly_array):
    row_coords = poly_array[1::2]
    col_coords = poly_array[0::2]
    row_min = np.amin(row_coords)
    col_min = np.amin(col_coords)
    col_max = np.amax(col_coords)
    row_max = np.amax(row_coords)
    return row_min, row_max, col_min, col_max


def create_tight_rect_around_locations(poly_arrays, im_shape, alpha=0.5):
    row_min_final, col_min_final = im_shape[0], im_shape[1]
    row_max_final, col_max_final = 0, 0
    for poly_array in poly_arrays:
        row_min, row_max, col_min, col_max = create_tight_rect_poly_array(poly_array)
        row_min_final = min(row_min_final, row_min)
        row_max_final = max(row_max_final, row_max)
        col_min_final = min(col_min_final, col_min)
        col_max_final = max(col_max_final, col_max)

    height = row_max_final - row_min_final
    width = col_max_final - col_min_final
    row_min = max(row_min_final - width * alpha, 0)
    col_min = max(col_min_final - height * alpha, 0)
    row_max = min(row_max_final + width * alpha, im_shape[0])
    col_max = min(col_max_final + height * alpha, im_shape[1])
    return int(row_min), int(row_max), int(col_min), int(col_max)


def draw_beautiful_polygon(poly_array, image_path, image_path_overlay, color_tuple, line_width=9):
    polygon_points = [(x,y) for x,y in  zip(poly_array[0::2], poly_array[1::2])]
    polygon_points.append(polygon_points[0])

    img = read_image(image_path)
    img_pil = Image.fromarray(img, 'RGB')
    dr = ImageDraw.Draw(img_pil)
    dr.line(polygon_points, fill=color_tuple, width=line_width)
    ellipse_thick = int(line_width/2)

    for point in polygon_points:
        dr.ellipse((point[0] - ellipse_thick, point[1] - ellipse_thick, point[0] + ellipse_thick, point[1] + ellipse_thick), fill=color_tuple)
    img_pil.save(image_path_overlay)








