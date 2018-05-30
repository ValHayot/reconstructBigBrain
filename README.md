# ReconstructBigBrain

A quick and dirty program that reconstructs the BigBrain into a NIfTI-1 image from the MINC blocks.

## Usage

`reconstruct.py <legend filename> <output filename> <folder containing the blocks> <block prefix> <block suffix> <numpy datatype>`

### Sample command

`reconstruct.py legend1000.mnc recon.nii ../blocks/ block40 inv.mnc np.ushort`
