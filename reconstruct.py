import nibabel as nib
import os
import h5py
import numpy as np
import math
import sys
import argparse
from time import time


def get_minc_data(filename):

    block_header = h5py.File(filename, 'r')

    minc_part = block_header['minc-2.0']
    # The whole image is the first of the entries in 'image'
    image = minc_part['image']['0']
    return (image['image'], minc_part)


def get_distance(block_1, block_2, dimension):

    dim1_names = block_1[0].attrs['dimorder'].decode("utf-8").split(',')
    dimensions1 = block_1[1]['dimensions']

    # voxel_dim @ 0, step @ 10, world @ 11
    dims1 = [dict(dimensions1[s].attrs.items()) for s in dim1_names]

    dim2_names = block_2[0].attrs['dimorder'].decode("utf-8").split(',')
    dimensions2 = block_2[1]['dimensions']

    dims2 = [dict(dimensions2[s].attrs.items()) for s in dim2_names]

    if dimension == 0:
        y1dim = dims1[0]['length']
        y1start = dims1[0]['start']
        y1step = dims1[0]['step']

        y2dim = dims2[0]['length']
        y2start = dims2[0]['start']
        y2step = dims2[0]['step']


        assert y1step == y2step

        return abs(y1start - y2start)/y2step + y2dim

    elif dimension == 1:
        z1dim = dims1[1]['length']
        z1start = dims1[1]['start']
        z1step = dims1[1]['step']

        z2dim = dims2[1]['length']
        z2start = dims2[1]['start']
        z2step = dims2[1]['step']

        assert z1step == z2step

        return abs(z1start - z2start)/z2step + z2dim

    elif dimension == 2:
        x1dim = dims1[2]['length']
        x1start = dims1[2]['start']
        x1step = dims1[2]['step']

        x2dim = dims2[2]['length']
        x2start = dims2[2]['start']
        x2step = dims2[2]['step']

        assert x1step == x2step

        return abs(x1start - x2start)/x2step + x2dim


def get_block_fn_from_id(block_id, block_folder, block_prefix="block40",
                         block_suffix="inv.mnc"):

    if block_suffix == ".mnc":
        filename = "{0}-{1}{2}" \
                   .format(block_prefix, str(block_id).zfill(4), block_suffix)
    else:
        filename = "{0}-{1}-{2}" \
                   .format(block_prefix, str(block_id).zfill(4), block_suffix)
    return os.path.join(block_folder, filename)


def create_header(legend, block_folder, filename, block_prefix, block_suffix):

    # get image dimensions by comparing distance between edge blocks
    y_blocks = [(get_minc_data(get_block_fn_from_id(el1, block_folder,
                                                    block_prefix,
                                                    block_suffix)),
                 get_minc_data(get_block_fn_from_id(el2, block_folder,
                                                    block_prefix,
                                                    block_suffix)))
                for el1, el2 in
                zip(set(legend[0, :, :].flatten()),
                    set(legend[-1, :, :].flatten()))]

    z_blocks = [(get_minc_data(get_block_fn_from_id(el1, block_folder,
                                                    block_prefix,
                                                    block_suffix)),
                 get_minc_data(get_block_fn_from_id(el2, block_folder,
                                                    block_prefix,
                                                    block_suffix)))
                for el1, el2 in
                zip(set(legend[:, 0, :].flatten()),
                    set(legend[:, -1, :].flatten()))]

    x_blocks = [(get_minc_data(get_block_fn_from_id(el1, block_folder,
                                                    block_prefix,
                                                    block_suffix)),
                 get_minc_data(get_block_fn_from_id(el2, block_folder,
                                                    block_prefix,
                                                    block_suffix)))
                for el1, el2 in
                zip(set(legend[:, :, 0].flatten()),
                    set(legend[:, :, -1].flatten()))]

    max_y = max(map(lambda x: get_distance(x[0], x[1], 0), y_blocks))
    max_z = max(map(lambda x: get_distance(x[0], x[1], 1), z_blocks))
    max_x = max(map(lambda x: get_distance(x[0], x[1], 2), x_blocks))
    print("Actual reconstructed image's actual dimensions: ",
          max_y, max_z, max_x)

    # Forced padding to enable 5x5x5 blocks
    # TODO: improve flexibility with padding

    pad_mul = 5

    print("Zero padding to ensure image can be broken down "
          "into even-sized blocks...")

    y_dim = int(max_y + pad_mul - (max_y % pad_mul))
    z_dim = int(max_z + pad_mul - (max_z % pad_mul))
    x_dim = int(max_x + pad_mul - (max_x % pad_mul))

    print("Padded image dimensions", y_dim, z_dim, x_dim)

    print("Image will be able to be deconstructed into blocks of size",
          y_dim / pad_mul, z_dim / pad_mul, x_dim / pad_mul)

    first_block = nib.load(get_block_fn_from_id(legend[0, 0, 0], block_folder, block_prefix, block_suffix))

    # TODO: improve header
    reconstructed_hdr = nib.Nifti1Header.from_header(first_block.header)
    reconstructed_hdr.set_sform(first_block.get_affine())
    reconstructed_hdr.set_data_shape((y_dim, z_dim, x_dim))

    # writing header to file
    print("Writing header to file")
    with open(filename, 'wb') as header:
        reconstructed_hdr.write_to(header)

    (x_start, y_start, z_start, _) = reconstructed_hdr.get_sform()[:, -1]

    return reconstructed_hdr.single_vox_offset, (y_dim, z_dim, x_dim)


def reconstruct(legend_fn, reconstructed_fn, block_folder, block_prefix,
                block_suffix, bytes_per_voxel, dtype):

    # assumes file ids are short
    legend = nib.load(legend_fn).get_data().astype(np.ushort)

    header_size, bb_dim = create_header(legend, block_folder, reconstructed_fn,
                                        block_prefix, block_suffix)

    blocks_copied = {}

    bb_ydim = int(bb_dim[0])
    bb_zdim = int(bb_dim[1])
    bb_xdim = int(bb_dim[2])

    print('Reconstructing with dimensions: ', bb_ydim, bb_zdim, bb_xdim)
    first_block = get_minc_data(
                                get_block_fn_from_id(
                                                     legend[0, 0, 0],
                                                     block_folder,
                                                     block_prefix,
                                                     block_suffix
                                                    )
                                )

    with open(reconstructed_fn, "r+b") as reconstructed:
        for x in range(0, legend.shape[2]):
            for z in range(0, legend.shape[1]):
                for y in range(0, legend.shape[0]):

                    block_filename = get_block_fn_from_id(
                                                          legend[y, z, x],
                                                          block_folder,
                                                          block_prefix,
                                                          block_suffix
                                                         )

                    if block_filename in blocks_copied:
                        continue
                    else:
                        blocks_copied[block_filename] = 1

                    block_img = nib.load(block_filename)
                    header = block_img.header
                    shape = header.get_data_shape()

                    ydim = shape[0]
                    zdim = shape[1]
                    xdim = shape[2]

                    block_data = block_img.get_data().astype(dtype)

                    minc_block = get_minc_data(block_filename)

                    y_block = int(get_distance(first_block, minc_block, 0) -
                                  ydim)
                    z_block = int(get_distance(first_block, minc_block, 1) -
                                  zdim)
                    x_block = int(get_distance(first_block, minc_block, 2) -
                                  xdim)

                    print('Writing block :', legend[y, z, x], ', Shape: ',
                          ydim, zdim, xdim)
                    print('Start of block :', y_block, z_block, x_block)

                    # Write to file
                    t = time()
                    for i in range(0, xdim):
                        for j in range(0, zdim):
                            reconstructed.seek(header_size + bytes_per_voxel*(
                                y_block + (z_block + j)*bb_ydim +
                                (x_block + i)*bb_ydim*bb_zdim), 0)
                            reconstructed.write(block_data[:, j, i].tobytes())

                    print(block_filename, "\t\t\tWrite time: ", time()-t)

        expected_bytes = header_size + bytes_per_voxel * (bb_ydim * bb_zdim *
                                                          bb_xdim)

        remainder_padding = expected_bytes - reconstructed.tell()

        if remainder_padding > 0:
            print('Number of bytes remaining ', remainder_padding)
            print('Adding some final padding...')
            zero_pad = np.zeros(int(remainder_padding / 2),
                                dtype=np.ushort, order='F')

            reconstructed.write(zero_pad.tobytes())


if __name__ == "__main__":

    # sample command:
    # python reconstruct.py legend1000.mnc recon.nii \
    # ../blocks/ block40 inv.mnc np.ushort

    parser = argparse.ArgumentParser(description=("Reconstruct a nifti image "
                                                  "given blocks and a legend"))
    parser.add_argument('legend', type=str,
                        help="The legend image to be used for reconstruction")
    parser.add_argument('outputimg', type=str,
                        help=("The output nifti-1 "
                              "reconstructed image filename."))
    parser.add_argument('blockfldr', type=str,
                        help="The folder containing the blocks")
    parser.add_argument('blockprfx', type=str,
                        help="prefix of minc blocks")
    parser.add_argument('blocksffx', type=str, help="suffix of minc blocks")
    parser.add_argument('dtype', type=str,
                        help=("Numpy datatype"
                              "(np.int16, np.ushort, np.uint16, np.float32, "
                              "np.float64)."))

    args = parser.parse_args()

    legend = args.legend
    reconstructed_fn = args.outputimg

    block_folder = args.blockfldr

    bytes_per_voxel = 0

    if args.dtype == "np.int16":
        bytes_per_voxel = np.dtype(int16).itemsize
        dtype = np.int16
    elif args.dtype == "np.ushort":
        bytes_per_voxel = np.dtype(np.ushort).itemsize
        dtype = np.ushort
    elif args.dtype == "np.uint16":
        bytes_per_voxel = np.dtype(np.uint16).itemsize
        dtype = np.uint16
    elif args.dtype == "np.float32":
        bytes_per_voxel = np.dtype(np.float32).itemsize
        dtype = np.float32
    else:
        bytes_per_voxel = np.dtype(np.float64).itemsize
        dtype = np.float64

    reconstruct(legend, reconstructed_fn, block_folder, args.blockprfx,
                args.blocksffx, bytes_per_voxel, dtype)
