#!/usr/bin/env python3
"""Full R26 matrix plus explicitly matched acceptance/performance controls."""
from __future__ import annotations

import argparse
import itertools
import traceback

import runtime as rt


def cell(label: str, *, image: str = rt.IMAGE, tp: int = 4, dcp: int = 4, spec: str = 'mtp0', cache: str = 'vram', kv: str = 'fp8_ds_mla', extra_env: dict | None = None, extra_args: list[str] | None = None, conc: str = '1,4,8,16', contexts: str = '0,32k', duration: int = 30, repeats: int = 1) -> None:
    try:
        if not rt.boot(label, image=image, tp=tp, dcp=dcp, spec=spec, cache=cache, kv=kv, extra_env=extra_env, extra_args=extra_args):
            return
        for repeat in range(1, repeats + 1):
            result_label = label if repeats == 1 else f'{label}-repeat{repeat}'
            rt.bench(result_label, conc=conc, contexts=contexts, duration=duration)
    except Exception as error:
        rt.record_gate('matrix-cell:' + label, False, {'error': repr(error), 'traceback': traceback.format_exc()})
    finally:
        rt.stop()


def smoke() -> None:
    cell('smoke-r26-dcp4-fp8-mtp0', dcp=4, duration=10, conc='1', contexts='0')


def priority() -> None:
    # Same FP8 target cache, 4K token budget,16 NCCL channels and input cells.
    # R25 has the native BF16 proposal projection. Override rather than unset
    # on R26 because the image itself supplies the NVFP4-head default.
    arms = [
        ('r25-a', rt.R25_IMAGE, 'bf16'),
        ('r26-bf16-head', rt.IMAGE, 'bf16'),
        ('r26-nvfp4-head', rt.IMAGE, 'nvfp4'),
        ('r26-overlay-nvfp4-head', rt.OVERLAY_IMAGE, 'nvfp4'),
        ('r25-b', rt.R25_IMAGE, 'bf16'),
    ]
    for tag, image, head in arms:
        head_env = {'VLLM_GLM53_MTP_DRAFT_HEAD': head}
        native_args = None
        if image == rt.OVERLAY_IMAGE:
            head_env['FAIRNESS_ENGINE'] = 'none'
            native_args = ['--prefill-compute-share', '0.4']
        cell(f'acceptance-dcp4-mtp3-{tag}', image=image, dcp=4, spec='mtp3',
             extra_env=head_env, extra_args=native_args, conc='1,8', contexts='0,32k', duration=30, repeats=2)
    for tag, image, head in arms[:3]:
        cell(f'acceptance-dcp1-mtp3-{tag}', image=image, dcp=1, spec='mtp3',
             extra_env={'VLLM_GLM53_MTP_DRAFT_HEAD': head}, conc='1,8', contexts='0', duration=60, repeats=2)
    for tag, image in [('r26-a', rt.IMAGE), ('r25', rt.R25_IMAGE), ('r26-b', rt.IMAGE)]:
        cell(f'execution-dcp4-nospec-{tag}', image=image, conc='1,8,16', contexts='0', duration=30)
    for tag, image in [('r25', rt.R25_IMAGE), ('r26', rt.IMAGE)]:
        cell(f'acceptance-dcp4-dflash-{tag}', image=image, spec='dflash2', conc='1,8', contexts='0,32k', repeats=2)


def matrix() -> None:
    configs = []
    for dcp, kv, spec in itertools.product((1, 2, 4), ('fp8_ds_mla', 'nvfp4_ds_mla'), ('mtp0', 'mtp3', 'dflash2')):
        configs.append({'label': f'dcp{dcp}-vram-{kv.split("_")[0]}-{spec}', 'dcp': dcp, 'kv': kv, 'spec': spec})
    for kv, spec in itertools.product(('fp8_ds_mla', 'nvfp4_ds_mla'), ('mtp0', 'mtp3', 'dflash2')):
        configs.append({'label': f'dcp4-lmcache-{kv.split("_")[0]}-{spec}', 'cache': 'lmcache', 'kv': kv, 'spec': spec})
    for kv in ('fp8_ds_mla', 'nvfp4_ds_mla'):
        name = kv.split('_')[0]
        configs.extend([
            {'label': f'dcp4-native-{name}-mtp0', 'cache': 'native', 'kv': kv},
            {'label': f'dcp4-vram-{name}-mtp0-b12xkda', 'kv': kv, 'extra_env': {'GLM53_KDA_PREFILL_BACKEND': 'b12x'}},
            {'label': f'dcp4-vram-{name}-mtp0-ranklocal', 'kv': kv, 'extra_env': {'DCP_CKV_GATHER': '0'}},
        ])
    rt.save_json('matrix-plan.json', configs)
    for config in configs:
        cell(**config)


def tuning() -> None:
    # Isolate batch budget from the DMA switch before trying their combination.
    factors = [(4096, None), (8192, None), (12288, None), (16384, None), (4096, '512KB'), (12288, '512KB')]
    for budget, dma in factors:
        settings = {'MAX_NUM_BATCHED_TOKENS': str(budget)}
        if dma is not None:
            settings['VLLM_PCIE_DMA_MIN_BYTES'] = dma
        cell(f'tuning-mtp3-dcp4-budget{budget}-dma{dma or "default"}', spec='mtp3',
             extra_env=settings, conc='1,8', contexts='0,32k', repeats=2)
    # MTP5 is a distinct curiosity, not substituted into the MTP3 release claim.
    cell('tuning-mtp5-dcp4-fp8', spec='mtp5', conc='1,8', contexts='0,32k')
    # D-Rock's changes are a separate artifact with the same native packages.
    for spec in ('mtp0', 'mtp3', 'dflash2'):
        cell(f'overlay-dcp4-fp8-{spec}', image=rt.OVERLAY_IMAGE, spec=spec,
             extra_env={'FAIRNESS_ENGINE': 'none'}, extra_args=['--prefill-compute-share', '0.4'],
             conc='1,8,16', contexts='0,32k')


def tp2() -> None:
    # First attempt the requested1M limit. A failure is preserved; no hidden
    # context reduction or DCP fallback is allowed to turn it into a pass.
    for kv, dcp in itertools.product(('fp8_ds_mla', 'nvfp4_ds_mla'), (1, 2)):
        cell(f'tp2-dcp{dcp}-{kv.split("_")[0]}-mtp0-1m', tp=2, dcp=dcp, kv=kv,
             conc='1,4', contexts='0,32k')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--section', choices=('smoke', 'priority', 'matrix', 'tuning', 'tp2', 'all'), default='all')
    args = parser.parse_args()
    sections = {'smoke': smoke, 'priority': priority, 'matrix': matrix, 'tuning': tuning, 'tp2': tp2}
    try:
        for name, phase in sections.items():
            if args.section in (name, 'all'):
                rt.note('PHASE START ' + name)
                phase()
                rt.save_json(f'{name}-completed.json', {'phase': name, 'complete': True, 'meaning': 'All planned cells attempted; consult gates for actual pass/fail.'})
                rt.note('PHASE COMPLETE ' + name)
    finally:
        rt.stop()


if __name__ == '__main__':
    main()
