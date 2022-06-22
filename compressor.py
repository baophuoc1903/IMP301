import cv2
import numpy as np


def compressor(original_image, code):
    compress_image_array = []
    for pixel_row_index in range(original_image.shape[0]):
        pixel_row = []
        for pixel_column_index in range(original_image.shape[1]):
            pixel_row.append(str(code[original_image[pixel_row_index][pixel_column_index]]))
        compress_image_array.append(pixel_row)
    return compress_image_array


def byte_stream(red_stream_image, green_stream_image, blue_stream_image, red_seperator, green_seperator):
    byte_stream = ""

    for stream_row_index in range(len(red_stream_image)):
        for stream_column_index in range(len(red_stream_image[0])):
            byte_stream += str(red_stream_image[stream_row_index][stream_column_index])

    byte_stream += red_seperator

    for stream_row_index in range(len(green_stream_image)):
        for stream_column_index in range(len(green_stream_image[0])):
            byte_stream += str(green_stream_image[stream_row_index][stream_column_index])

    byte_stream += green_seperator

    for stream_row_index in range(len(blue_stream_image)):
        for stream_column_index in range(len(blue_stream_image[0])):
            byte_stream += str(blue_stream_image[stream_row_index][stream_column_index])

    return byte_stream


def decoder(bit_stream, red_stream_decoder, green_stream_decoder, blue_stream_decoder, img_path="./Original.jpeg"):
    code_stream = bit_stream_decode(img_path)
    # code_stream = []
    if code_stream == []:
        while code_search(bit_stream, red_stream_decoder, 1) != 'seperator':
            code = code_search(bit_stream, red_stream_decoder, 1)
            code_stream.append(red_stream_decoder[code])
            bit_stream = bit_stream.replace(code, '', 1)

        code = code_search(bit_stream, red_stream_decoder, 1)
        bit_stream = bit_stream.replace(code, '', 1)
        print('Red over, Green started')

        while code_search(bit_stream, green_stream_decoder, 1) != 'seperator':
            code = code_search(bit_stream, green_stream_decoder, 1)
            code_stream.append(green_stream_decoder[code])
            bit_stream = bit_stream.replace(code, '', 1)

        code = code_search(bit_stream, green_stream_decoder, 1)
        bit_stream = bit_stream.replace(code, '', 1)
        print('Green over, Blue started')

        while bit_stream != '':
            code = code_search(bit_stream, blue_stream_decoder, 1)
            code_stream.append(blue_stream_decoder[code])
            bit_stream = bit_stream.replace(code, '', 1)

    return code_stream


def code_search(small_bit_stream, search_dict, slicing_index):
    code = small_bit_stream[:slicing_index]
    if search_dict.get(code, None) == None:
        return code_search(small_bit_stream, search_dict, slicing_index + 1)
    else:
        return code


def bit_stream_decode(img_path):
    file_to_be_decoded = cv2.imread(img_path)
    file_to_be_decoded = cv2.cvtColor(file_to_be_decoded, cv2.COLOR_BGR2RGB)

    file_x, file_y, file_z = file_to_be_decoded.shape
    file_size = file_x * file_y * file_z

    decoded_stream = []
    for z in range(file_z):
        for x in range(file_x):
            for y in range(file_y):
                decoded_stream.append(file_to_be_decoded[x][y][z])

    return decoded_stream


def image_restorer(red_channel_image, green_channel_image, blue_channel_image):
    x_max, y_max = np.array(red_channel_image).shape

    restored_image = []

    for x in range(x_max):
        y_set = []
        for y in range(y_max):
            z_set = [red_channel_image[x][y], green_channel_image[x][y], blue_channel_image[x][y]]
            y_set.append(z_set)
        restored_image.append(y_set)

    return restored_image