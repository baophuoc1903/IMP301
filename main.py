import cv2
import numpy
import matplotlib.pyplot as plt
from block_threshold_mean import *
from dct import *
import histogram
import huffman
import json
import compressor


def huffman_encode(img):
    img_histogram = histogram.histogram_array_generator(img)
    img_histogram_probability_distribution = histogram.probability_distribution_generator(img_histogram,
                                                                                          img.shape[0] * img.shape[1])
    img_histogram_probability_distribution['separator'] = 0
    img_huffman_coding = huffman.Huffman_Coding(img_histogram_probability_distribution)

    img_coded_pixels, img_reverse_coded_pixels = img_huffman_coding.compress()

    with open('./encode.json', 'w') as fp:
        json.dump(img_coded_pixels, fp)
    with open('./decode.json', 'w') as fp:
        json.dump(img_reverse_coded_pixels, fp)

    compressed_image = compressor.compressor(img, img_coded_pixels)

    bit_stream = compressor.byte_stream(compressed_image, compressed_image, compressed_image,
                                        img_coded_pixels['separator'], img_coded_pixels['separator'])
    print('Compression ratio:', (len(bit_stream) / (img.shape[0]*img.shape[1] * 3 * 8)))

    with open('bit_stream.txt', 'w') as fp:
        fp.write(bit_stream)
    return img_coded_pixels, img_reverse_coded_pixels


def huffman_decode(img_decode_json='./decode.json', bit_stream_path='./bit_stream.txt'):
    img_decoder = json.load(open(img_decode_json, 'r'))
    with open(bit_stream_path, 'r') as fr:
        bit_stream = fr.read()
    pixel_stream = compressor.decoder(bit_stream, img_decoder, img_decoder, img_decoder)
    with open('image_pixel_stream.txt', 'w') as fr:
        fr.write(str(pixel_stream))


def restone_img(img_shape, gray=True):
    with open('image_pixel_stream.txt', 'r') as fr:
        pixel_stream = fr.read()
    pixel_stream = pixel_stream.replace('[', '')
    pixel_stream = pixel_stream.replace(']', '')
    pixel_stream = pixel_stream.split(', ')
    pixel_stream = [int(pixel) for pixel in pixel_stream]

    st_channel_pixel_stream = pixel_stream[:int(len(pixel_stream) / 3)]
    nd_channel_pixel_stream = pixel_stream[int(len(pixel_stream) / 3):int((2 * len(pixel_stream)) / 3)]
    rd_channel_pixel_stream = pixel_stream[int((2 * len(pixel_stream)) / 3):int(len(pixel_stream))]

    st_channel_pixel_stream = np.reshape(st_channel_pixel_stream, img_shape)
    nd_channel_pixel_stream = np.reshape(nd_channel_pixel_stream, img_shape)
    rd_channel_pixel_stream = np.reshape(rd_channel_pixel_stream, img_shape)

    if gray:
        return np.array(st_channel_pixel_stream)
    else:
        return np.array(compressor.image_restorer(st_channel_pixel_stream,
                                                  nd_channel_pixel_stream,
                                                  rd_channel_pixel_stream))


if __name__ == '__main__':
    # READ IMAGE
    img = cv2.imread(r"/Users/nguyenbaophuoc/Downloads/DIP3E_CH08_Original_Images/Fig0809(a).tif", 0)

    # BLOCK TRANSFORM USING DCT
    image_dct = fast_dct_manual(img, 8)
    bitmap, image_quantize = btc_manual(image_dct, 8, threshold=4/64)
    # image_quantize = (image_quantize//32) * 32  # 7 bit

    # Approximate original image
    img_back = idct_manual(image_quantize, 8).astype(int)
    dct_psnr(img, img_back)
    plot_save(img, 'Original.jpeg', size=6)
    plot_save(img_back, 'Decompression_4.jpeg', size=6)
    plot_save((img-img_back)**2, "Different_4.jpeg", size=6)
