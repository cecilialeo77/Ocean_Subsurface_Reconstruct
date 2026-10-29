"""顺序执行IPRC预训练和GLORYS微调；默认不运行最终测试。"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from .config import load_config
from .transfer import stage_name, validate_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrain-config', required=True)
    parser.add_argument('--finetune-config', required=True)
    parser.add_argument('--max-samples', type=int, help='试跑：每个划分最多使用的月份数')
    parser.add_argument('--epochs', type=int, help='试跑时两个阶段各自的总轮数')
    parser.add_argument('--run-test', action='store_true', help='两个阶段完成后显式运行最终GLORYS测试')
    args = parser.parse_args()
    pre, fine = load_config(args.pretrain_config), load_config(args.finetune_config)
    for cfg in (pre, fine): validate_stage(cfg)
    if stage_name(pre) != 'iprc_pretrain' or stage_name(fine) != 'glorys_finetune':
        parser.error('Expected iprc_pretrain and glorys_finetune configs')
    if pre['model'] != fine['model']:
        parser.error('Two stages must use identical model configuration')
    if Path(pre['training']['output_dir']).resolve() == Path(fine['training']['output_dir']).resolve():
        parser.error('Two stages must use different output directories')
    if args.max_samples is not None and args.run_test:
        parser.error('Do not combine a smoke run with final testing; use explicit preview reconstruction')
    common = []
    if args.max_samples is not None: common += ['--max-samples', str(args.max_samples)]
    if args.epochs is not None: common += ['--epochs', str(args.epochs)]
    suffix = Path('smoke') if args.max_samples is not None else Path('.')
    pre_output = Path(pre['training']['output_dir']) / suffix
    fine_output = Path(fine['training']['output_dir']) / suffix
    def run(module, *extra):
        command = [sys.executable, '-m', f'glorys_swin_reconstruction.{module}', *extra]
        print('执行：' + subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, check=True)
    run('train', '--config', args.pretrain_config, *common)
    run('train', '--config', args.finetune_config,
        '--init-checkpoint', str(pre_output/'best.pt'), *common)
    if args.run_test:
        run('reconstruct', '--config', args.finetune_config,
            '--checkpoint', str(fine_output/'best.pt'), '--split', 'test')
    print('两阶段训练完成。断点续训请分别使用 train --resume，不要重新执行整个流程。')


if __name__ == '__main__':
    main()
