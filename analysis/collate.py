"""Offline collation: results/<host>/{bench.csv,*.monitor.csv,sysinfo.csv}
-> results/summary.csv, results/sysinfo.md, figures/*.png

usage: python3 analysis/collate.py
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)

# colour-blind-safe pair, one colour per host
COLOR = {'orangepi': '#d95f02', 'oiseau': '#1b9e77'}
LABEL = {'orangepi': 'Orange Pi 5 (RK3588)', 'oiseau': 'Intel N150'}

bench = pd.concat([pd.read_csv(f).assign(host=os.path.basename(os.path.dirname(f)))
                   for f in glob.glob(os.path.join(RES, '*', 'bench.csv'))])
bench['label'] = bench.host.map(LABEL) + '\n' + bench.engine + ' ' + bench.device


def monitor(host, test):
    return pd.read_csv(os.path.join(RES, host, f'{test}.monitor.csv'))


# ---- summary table -------------------------------------------------------
g = bench.groupby(['host', 'test', 'engine', 'device', 'threads', 'batch'])
summary = g.latency_ms.agg(n='count', mean='mean', median='median',
                           p95=lambda s: s.quantile(.95),
                           p99=lambda s: s.quantile(.99), std='std').reset_index()
# median, not mean: other services on the hosts add a heavy upper tail
summary['inputs_per_s'] = 1000 * summary.batch / summary['median']
# BirdNET: one call = 3 s of audio
bn = summary.engine.str.startswith('birdnet')
summary['realtime_factor'] = None
summary.loc[bn, 'realtime_factor'] = 3000 / summary.loc[bn, 'median']

# mean CPU% / NPU% over each test's busy period (excludes idle lead/tail)
extra = []
for _, r in summary.iterrows():
    m = monitor(r.host, r.test)
    dur = bench[(bench.host == r.host) & (bench.test == r.test)].t_s.max()
    busy = m[(m.t_s >= 5) & (m.t_s <= 5 + dur)]
    extra.append({'cpu_pct': busy.cpu_pct.mean(),
                  # only one NPU core is used by this model: report the busiest
                  'npu_pct': busy[['npu0_pct', 'npu1_pct', 'npu2_pct']].mean().max(),
                  'temp_max_c': busy.temp_c.max(), 'cpu_mhz': busy.cpu_mhz.mean()})
summary = pd.concat([summary, pd.DataFrame(extra)], axis=1)
summary.round(2).to_csv(os.path.join(RES, 'summary.csv'), index=False)

# ---- sysinfo tables for the post ----------------------------------------
with open(os.path.join(RES, 'sysinfo.md'), 'w') as f:
    for p in sorted(glob.glob(os.path.join(RES, '*', 'sysinfo.csv'))):
        host = os.path.basename(os.path.dirname(p))
        f.write(f'**{host}**\n\n| Item | Value |\n| :--- | :--- |\n')
        for _, r in pd.read_csv(p, keep_default_na=False).iterrows():
            f.write(f'| {r.key} | {r.value} |\n')
        f.write('\n')


def style(ax, title, xl, yl):
    ax.set_title(title, loc='left', fontsize=11)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.25)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)


# ---- 1. latency distribution: ResNet-18, batch 1, best config per engine --
r18 = bench[(bench.batch == 1) & bench.engine.str.contains('rknn|torch|openvino')
            & ~bench.test.str.contains('sustained|^yolo')]
best = (r18.groupby(['host', 'test']).latency_ms.median().reset_index()
        .sort_values('latency_ms').groupby('host').head(3))
sel = r18.merge(best[['host', 'test']])
fig, ax = plt.subplots(figsize=(8, 4.5))
tests = sel.groupby(['host', 'test']).latency_ms.median().sort_values().index.tolist()
ax.boxplot([sel[(sel.host == h) & (sel.test == t)].latency_ms for h, t in tests],
           tick_labels=[f'{LABEL[h].split(" (")[0]}\n{t}' for h, t in tests],
           showfliers=False, patch_artist=True,
           boxprops=dict(facecolor='none'), medianprops=dict(color='black'))
ax.set_yscale('log')
style(ax, 'ResNet-18 single-image latency (3 fastest configs per host)', '', 'ms per inference (log)')
ax.tick_params(axis='x', labelsize=7)
save(fig, 'latency_distribution.png')

# ---- 2. sustained runs: latency + temperature + frequency over time -------
sus = [t for t in bench.test.unique() if 'sustained' in t]
fig, axes = plt.subplots(3, len(sus), figsize=(4.5 * len(sus), 7), sharex='col', squeeze=False)
for j, t in enumerate(sus):
    for host in bench[bench.test == t].host.unique():
        d = bench[(bench.host == host) & (bench.test == t)]
        roll = d.set_index('t_s').latency_ms.rolling(50, min_periods=1).median()
        m = monitor(host, t)
        m = m.assign(t_s=m.t_s - 5)
        axes[0, j].plot(roll.index, roll.values, color=COLOR[host], label=LABEL[host])
        axes[1, j].plot(m.t_s, m.temp_c, color=COLOR[host])
        axes[2, j].plot(m.t_s, m.cpu_mhz, color=COLOR[host])
    style(axes[0, j], t, '', 'latency ms (rolling median)')
    style(axes[1, j], '', '', 'max temp °C')
    style(axes[2, j], '', 'seconds', 'mean CPU MHz')
    axes[0, j].legend(frameon=False, fontsize=8)
save(fig, 'sustained.png')

# ---- 3. CPU vs NPU utilisation while running inference --------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
u = summary[summary.engine.str.contains('rknn|torch|openvino') & (summary.batch == 1)
            & ~summary.test.str.contains('sustained|^yolo')].copy()
u['name'] = u.host.map(lambda h: LABEL[h].split(' (')[0]) + '\n' + u.test
u = u.sort_values(['host', 'test'])
x = range(len(u))
ax.bar([i - .2 for i in x], u.cpu_pct, .4, label='CPU (all cores)', color='#7570b3')
ax.bar([i + .2 for i in x], u.npu_pct.fillna(0), .4, label='NPU (busiest core)', color='#d95f02')
ax.set_xticks(list(x))
ax.set_xticklabels(u.name, fontsize=6, rotation=60, ha='right')
style(ax, 'Utilisation while running ResNet-18', '', '% busy')
ax.legend(frameon=False)
save(fig, 'utilisation.png')

# ---- 4. scaling with threads / batch --------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
t = summary[summary.test.str.startswith('torch_threads')]
for host, d in t.groupby('host'):
    axes[0].plot(d.threads, d.inputs_per_s, 'o-', color=COLOR[host], label=LABEL[host])
style(axes[0], 'PyTorch CPU: thread scaling (batch 1)', 'threads', 'images / s')
axes[0].legend(frameon=False)
b = summary[summary.test.str.match(r'torch_(batch_|threads_(4|8)$)')
            & (summary.engine == 'torch-fp32')]
b = b[b.threads == b.groupby('host').threads.transform('max')]
for host, d in b.groupby('host'):
    d = d.sort_values('batch')
    axes[1].plot(d.batch, d.inputs_per_s, 'o-', color=COLOR[host], label=LABEL[host])
style(axes[1], 'PyTorch CPU: batch size (all cores)', 'batch', 'images / s')
save(fig, 'scaling.png')

# ---- 5. BirdNET real-time factor ------------------------------------------
bd = summary[summary.engine.str.startswith('birdnet')]
if len(bd):
    fig, ax = plt.subplots(figsize=(7, 4))
    for host, d in bd.groupby('host'):
        ax.plot(d.threads, d.realtime_factor.astype(float), 'o-', color=COLOR[host], label=LABEL[host])
    ax.axhline(1, color='grey', ls='--', lw=1)
    ax.set_yscale('log')
    style(ax, 'BirdNET V2.4 (TFLite FP32, CPU): audio seconds per wall second',
          'threads', 'real-time factor (log)')
    ax.legend(frameon=False)
    save(fig, 'birdnet.png')

# ---- 6. YOLOv5s: frames per second, single stream and concurrent streams ---
yo = summary[summary.test.str.startswith('yolo')].copy()
if len(yo):
    yo['streams'] = yo.test.str.extract(r'streams_(\d+)$')[0].fillna(1).astype(int)
    yo['stage'] = yo.test.str.extract(r'_(infer|e2e)')[0]
    # aggregate fps over all concurrent processes: calls completed / timed seconds
    agg = {}
    for (h, t), d in bench[bench.test.str.startswith('yolo')].groupby(['host', 'test']):
        agg[(h, t)] = len(d) / d.t_s.max()
    yo['fps'] = [agg[(r.host, r.test)] if r.streams > 1 else r.inputs_per_s
                 for r in yo.itertuples()]
    yo['engine_dev'] = yo.host.map(lambda h: LABEL[h].split(' (')[0]) + ' ' + yo.engine.str.replace('yolov5s-', '') \
        + ' ' + yo.device
    yo.round(2).to_csv(os.path.join(RES, 'yolo_summary.csv'), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [3, 2]})
    one = yo[yo.streams == 1].sort_values('fps')
    for stage, off, hatch in [('infer', -.2, ''), ('e2e', .2, '//')]:
        d = one[one.stage == stage]
        names = list(dict.fromkeys(one.engine_dev))
        xs = [names.index(n) + off for n in d.engine_dev]
        axes[0].bar(xs, d.fps, .4, hatch=hatch, label={'infer': 'model only', 'e2e': 'with letterbox + decode + NMS'}[stage],
                    color=[COLOR[h] for h in d.host], edgecolor='white')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, fontsize=7, rotation=30, ha='right')
    style(axes[0], 'YOLOv5s 640x640, one stream', '', 'frames / s')
    axes[0].legend(frameon=False, fontsize=8)
    for (h, e), d in yo[yo.stage == 'e2e'].groupby(['host', 'engine_dev']):
        if (d.streams > 1).any():
            d = d.sort_values('streams')
            axes[1].plot(d.streams, d.fps, 'o-', label=e, color=COLOR[h],
                         ls='-' if 'npu' in e else '--')
    style(axes[1], 'Aggregate fps, concurrent streams (end to end)', 'concurrent processes', 'frames / s')
    axes[1].legend(frameon=False, fontsize=7)
    save(fig, 'yolo.png')

print(summary.round(2).to_string())
