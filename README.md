# Multi-Dynamic Deep Image Prior (M-DIP)

This repository is a clone of the original M-DIP codebase with additional support for radial acquisition patterns, including radial train-data loading and radial data-consistency integration.

Contact: marc.vornehm@fau.de, rizwan.ahmad@osumc.edu

## Publication
Please cite the [original publication](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.70000) as follows:

> Vornehm M, Chen C, Sultan MA, et al. Multi-dynamic deep image prior for cardiac MRI. *Magn Reson Med.* 2025; 94(6): 2668-2679. doi: 10.1002/mrm.70000

Additionally, a preprint is available on [arXiv](https://arxiv.org/abs/2412.04639).

## Overview
Multi-Dynamic Deep Image Prior is a novel unsupervised reconstruction framework for accelerated realtime cardiac MRI.
M-DIP employs a spatial dictionary to synthesize a time-dependent template image, which is further refined using time-dependent deformation fields that model cardiac and respiratory motion.

![M-DIP overview](overview.png)

## Setup
M-DIP is tested using Python 3.12 on Linux. Install the required packages using
> `pip install -r requirements.txt`

To run LR-DIP for comparison, create a separate environment based on Python 3.12 and install the respective packages using
> `pip install -r requirements_lrdip.txt`

## Data
Raw data should be placed in a subdirectory `data/`.
The in-vivo data used in our [publication](#publication) cannot be published.
Instead, we recommend using real-time cine data from the [OCMR dataset](https://ocmr.info), ideally with at least 100 frames.

For radial multi-slice cardiac cine experiments, organize preprocessed train data as one `.npz` file per slice or view. Each file should contain radial k-space samples, trajectory/bin information, and coil sensitivity maps using the following fields:

- `Y_data`: complex radial k-space samples with shape `[spoke, coil, readout, 1]`.
- `X_data`: radial sample metadata with shape `[spoke, 2]`, where column 0 is the spoke angle in radians and column 1 is the normalized temporal bin.
- `csm`: complex coil sensitivity maps with shape `[coil, x, y]`.
- `recon_fs`, `recon_sense`, or similar optional reference reconstructions with shape `[x, y, frame]`.
- `hollow_mask`: optional spatial mask with shape `[x, y]`.

A typical directory layout is:

```text
data/
  <dataset_name>/
    traindata/
      slice_1_8_nbins30.npz
      slice_2_8_nbins30.npz
    raw/
    recons/
```

The radial notebook expects `raw_folder` to point to the `traindata/` directory and `filename` to select the slice-specific `.npz` file.

## Implemented methods
- M-DIP: in `M-DIP.ipynb`
- Low rank plus sparse (Otazo et al., https://doi.org/10.1002/mrm.25240): in `M-DIP.ipynb`, section `L+S reconstruction`.
- LR-DIP (Hamilton et al., https://doi.org/10.1007/s10334-023-01088-w): in `LR-DIP.ipynb`, see more details below.

## M-DIP
### Running in Jupyter Notebook
All parameters and paths can be changed via the variables defined in the second cell. Specify the folder containing raw data in MRD format in `raw_folder`, the desired output folder in `out_folder`, and the name of the raw data file in `filename`.

### Running from command line
The Jupyter Notebook can be executed from the command line using the papermill package (included in `requirements.txt`).

You can furthermore overwrite any of the parameters defined in the second cell using the `-p` switch, where each parameter has to be preceeded by the switch:
> `papermill M-DIP.ipynb temp.ipynb [-p <parameter> <value> ...]`

Example:
> `papermill M-DIP.ipynb temp.ipynb -p n_bases 15 -p filename <filename>`

Note that `temp.ipynb` can be any arbitrary Jupyter Notebook filename.

## LR-DIP
Code for the LR-DIP reconstruction was kindly provided by Jesse Hamilton (University of Michigan) and was adjusted for Cartesian data and for parameterization using papermill.
Note that the code does not follow the descriptions from the paper in every aspect. However, the provided code gave the best results.

Similar to the procedure described for M-DIP above, the notebook can be executed via the command line using
> `papermill LR-DIP.ipynb temp.ipynb [-p <parameter> <value> ...]`
