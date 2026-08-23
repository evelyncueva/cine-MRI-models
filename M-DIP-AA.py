#!/usr/bin/env python
"""Run radial M-DIP on preprocessed multi-slice cardiac cine .npz data."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault('NUMBA_CACHE_DIR', f'/tmp/numba_cache_{os.environ.get("USER", "condor")}')
os.environ.setdefault('MPLCONFIGDIR', f'/tmp/mpl_cache_{os.environ.get("USER", "condor")}')
Path(os.environ['NUMBA_CACHE_DIR']).mkdir(parents=True, exist_ok=True)
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from dip import evaluate, models, mri, plotting
from dip.dataset import RadialTrainDataset
from dip.mdip import MDIP


def parse_args():
    parser = argparse.ArgumentParser(description='Run radial M-DIP for train-data .npz files.')
    parser.add_argument('--raw-folder', default='./data/AA/traindata', help='Folder containing radial .npz files.')
    parser.add_argument('--out-folder', default='./results/AA', help='Output folder.')
    parser.add_argument('--filename', default='slice_1_8_nbins30.npz', help='Input .npz filename.')
    parser.add_argument('--slice-idx', type=int, default=0, help='Slice index. Radial train files usually contain one slice.')
    parser.add_argument('--n-iter', type=int, default=100, help='Optimization iterations.')
    parser.add_argument('--save-every', type=int, default=0, help='Save intermediate results every N iterations.')
    parser.add_argument('--batch-size', type=int, default=96, help='Temporal batch size.')
    parser.add_argument('--cuda-num', type=int, default=0, help='CUDA device index.')
    parser.add_argument('--cpu', action='store_true', help='Force CPU execution.')
    parser.add_argument('--no-flow', action='store_true', help='Disable motion/deformation flow.')
    parser.add_argument('--activate-flow-after', type=int, default=0, help='Iteration to activate flow.')
    parser.add_argument('--n-bases', type=int, default=16, help='Number of spatial bases.')
    parser.add_argument('--zs-chans', type=int, default=2, help='Static code vector channels.')
    parser.add_argument('--zt-chans', type=int, default=4, help='Temporal code vector channels.')
    parser.add_argument('--p-dropout', type=float, default=0)
    parser.add_argument('--noise-reg', type=float, default=0.05)
    parser.add_argument('--lr-max', type=float, default=1e-3)
    parser.add_argument('--lr-min', type=float, default=1e-6)
    parser.add_argument('--lr-static-factor', type=float, default=1)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--lambda-flow-spatial', type=float, default=0.10)
    parser.add_argument('--lambda-flow-temporal', type=float, default=0.05)
    parser.add_argument('--lambda-zt', type=float, default=0)
    parser.add_argument('--lambda-basis', type=float, default=0)
    parser.add_argument('--ksp-scale', type=float, default=100)
    parser.add_argument('--monitor-every', type=int, default=50, help='Metric interval when reference data is present.')
    parser.add_argument(
        '--radial-operator', choices=('nufft', 'grid'), default='nufft',
        help='Radial data-consistency operator. Use nufft for reconstruction quality; grid is a fast approximation.',
    )
    return parser.parse_args()


def build_output_path(out_folder: str, filename: str, slice_idx: int) -> Path:
    filename_fid = re.search(r'FID\d*', Path(filename).stem)
    if filename_fid is not None:
        dset_name = filename_fid.group(0)
    else:
        dset_name = Path(filename).stem.split('_')[-1]
    output_path = Path(out_folder) / f'{dset_name}' / f'slice_{slice_idx:02d}'
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_loss_plot(mdip: MDIP):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.semilogy(mdip.metrics['total_loss'], linewidth=0.6, label='$L$')
    plt.semilogy(mdip.metrics['kspace_loss'], linewidth=0.6, label='$L_k$')
    if mdip.lambda_flow_spatial > 0:
        plt.semilogy(mdip.metrics['flow_loss_spatial'], linewidth=0.6, label='$L_{def,s}$')
    if mdip.lambda_flow_temporal > 0:
        plt.semilogy(mdip.metrics['flow_loss_temporal'], linewidth=0.6, label='$L_{def,t}$')
    if mdip.lambda_zt > 0:
        plt.semilogy(mdip.metrics['zt_loss'], linewidth=0.6, label='$L_{zt}$')
    if mdip.lambda_basis > 0:
        plt.semilogy(mdip.metrics['basis_loss'], linewidth=0.6, label='$L_b$')
    plt.legend()
    plt.title('Loss')
    plt.subplot(122)
    plt.semilogy(mdip.metrics['residual'], linewidth=0.6)
    plt.title('Residual')
    plt.tight_layout()
    plt.savefig(mdip.output_path / 'loss.png')
    plt.close()


def main():
    args = parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    dtype = torch.float32
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.cuda_num}')
    print(f'Using device: {device}')

    input_path = Path(args.raw_folder) / args.filename
    if not input_path.exists():
        raise FileNotFoundError(
            f'Input file not found: {input_path.resolve()}\n'
            'Copy the radial .npz file to this path, or pass the correct folder with '
            '`--raw-folder /path/to/traindata` or `RAW_FOLDER=/path/to/traindata ./run_mdip_condor.sh ...`.'
        )
    if not RadialTrainDataset.can_load(input_path):
        raise ValueError(
            f'Unsupported radial train-data file: {input_path.resolve()}\n'
            'Expected a .npz file containing Y_data, X_data, and csm.'
        )

    output_path = build_output_path(args.out_folder, args.filename, args.slice_idx)
    data = RadialTrainDataset(input_path)
    data.crop_readout_oversampling()
    data.whiten()
    data.sl = args.slice_idx

    print(f'Number of slices: {data.n_slices}')
    print(f'Number of frames: {data.n_phases}')
    print(f'Number of coils:  {data.n_coils}')
    print(f'Matrix size:      {data.matrix_size}')

    k = data.k
    m = data.m.astype(np.int8)[:, 0]
    sens_maps = data.estimate_sens_maps(k)
    trajectory = data.trajectory_for_slice

    img_avg = data.coil_images(k)
    plotting.plot_multichannel(
        img_avg, channel_axis=0, columns=6, figheight_per_row=3, figsize=(10, None),
        complex='abs', save_path=output_path / 'coils_compressed.png', show=False, cmap='gray',
    )
    plotting.plot_multichannel(
        sens_maps, channel_axis=0, columns=6, figheight_per_row=3, figsize=(10, None), complex='abs',
        save_path=output_path / 'sens_maps.png', show=False, cmap='gray',
    )

    basis_gen = models.UNet(
        enc_channels=[args.zs_chans, 32, 64, 64, 64],
        dec_channels=[64, 64, 64, 32, 16],
        out_channels=2 * args.n_bases,
        kernel_size=3,
        n_convs_per_block=2,
        p_dropout=args.p_dropout,
        interpolation_mode='bilinear',
    ).to(dtype=dtype)

    coeff_gen = None
    if args.n_bases > 1:
        coeff_gen = models.MLP(
            feature_lengths=[args.zt_chans, 32, 64, 128, 256, 128, 64, 2 * args.n_bases],
            last_activation=False,
            p_dropout=args.p_dropout,
        ).to(dtype=dtype)

    code_vector_size = basis_gen.required_input_size(data.matrix_size, 2)
    unet_bottleneck_size = basis_gen.get_bottleneck_size(code_vector_size)
    flow_gen = models.FlowGenerator(
        mlp_features=[args.zt_chans, 32, 64, 64, 64],
        conv_input_size=unet_bottleneck_size,
        conv_channels=[64, 64, 64, 64, 64],
        n_convs_per_block=3,
        p_dropout=args.p_dropout,
        interpolation_mode='nearest',
    ).to(dtype=dtype)

    output_size = basis_gen.get_output_size(code_vector_size)
    transformer = models.SpatialTransformer(output_size).to(dtype=dtype)

    k_tor = torch.from_numpy(k)
    k_max = torch.max(torch.abs(k_tor)).item()
    k_tor = k_tor * args.ksp_scale / k_max
    k_tor = k_tor.to(dtype=torch.promote_types(dtype, torch.complex32))

    m_tor = torch.from_numpy(m)[:, None].to(dtype=dtype)
    sen_tor = torch.from_numpy(sens_maps).to(dtype=torch.promote_types(dtype, torch.complex32))
    trajectory_tor = torch.from_numpy(trajectory).to(dtype=dtype)

    zs = torch.empty(1, args.zs_chans, *code_vector_size, dtype=dtype).uniform_(0, 0.1)
    zt = torch.zeros(data.n_phases, args.zt_chans, dtype=dtype)

    params = vars(args).copy()
    with open(output_path / 'params.yaml', 'w') as f:
        yaml.dump_all(
            [
                {'params': params},
                {'basis_gen': basis_gen.config},
                {'coeff_gen': coeff_gen.config if coeff_gen is not None else None},
                {'flow_gen': flow_gen.config},
            ],
            f,
            explicit_start=True,
            default_flow_style=False,
        )

    mdip = MDIP(
        zs=zs,
        zt=zt,
        basis_gen=basis_gen,
        coeff_gen=coeff_gen,
        flow_gen=flow_gen,
        transformer=transformer,
        matrix_size=data.matrix_size,
        n_frames=data.n_phases,
        imaging_fs=1000 / data.tres,
        lambda_flow_spatial=args.lambda_flow_spatial,
        lambda_flow_temporal=args.lambda_flow_temporal,
        lambda_zt=args.lambda_zt,
        lambda_basis=args.lambda_basis,
        noise_reg=args.noise_reg,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        lr_static_factor=args.lr_static_factor,
        weight_decay=args.weight_decay,
        output_path=output_path,
    ).to_device(device)

    has_reference = data.ground_truth is not None
    mdip.optimize(
        k=k_tor.to(device=device),
        sens=sen_tor.to(device=device),
        mask=m_tor.to(device=device),
        n_iter=args.n_iter,
        save_every=args.save_every,
        activate_flow_after=(args.n_iter if args.no_flow else args.activate_flow_after),
        batch_size=args.batch_size,
        monitor_every=args.monitor_every if has_reference else -1,
        monitor_gt=data.ground_truth[args.slice_idx] * args.ksp_scale / k_max if has_reference else None,
        trajectory=trajectory_tor.to(device=device),
        radial_operator=args.radial_operator,
    )
    mdip.save()
    save_loss_plot(mdip)
    if has_reference:
        mdip.save_metrics()

    with mdip.no_grad_and_eval():
        cine, basis, coeffs, flow = mdip.forward_(generate_flow=not args.no_flow)
        cine = cine.cpu().numpy() / args.ksp_scale * k_max
        basis = basis.cpu().numpy()
        coeffs = coeffs.cpu().numpy()
        flow = flow.cpu().numpy()

    cine = mri.center_crop(cine, data.recon_size, (1, 2))
    mdip.save_cine(cine, equalize_histogram=True)
    mdip.save_basis(basis)
    if args.n_bases > 1:
        mdip.save_coeffs(coeffs)
    mdip.save_flow(flow)
    mdip.save_static_code_vector()
    mdip.save_temporal_code_vector()

    if has_reference:
        cine_gt = data.ground_truth[args.slice_idx]
        metrics = evaluate.get_metrics(np.abs(cine_gt), np.abs(cine))
        with open(output_path / 'metrics_final.yaml', 'w') as f:
            yaml.safe_dump(metrics.to_dict(orient='index'), f)

    print(f'Finished. Outputs saved to {output_path}')


if __name__ == '__main__':
    main()
