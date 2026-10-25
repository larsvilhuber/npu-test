"""Shared timing loop and CSV writer for the benchmarks.

Output columns: test,engine,device,threads,batch,iter,t_s,latency_ms
  t_s         seconds since the first timed iteration started
  latency_ms  wall time of one call (a call processes `batch` inputs)
"""
import argparse
import csv
import time


def add_args(p):
    p.add_argument('--test', required=True, help='label written to the CSV')
    p.add_argument('--out', required=True, help='CSV path (appended to)')
    p.add_argument('--iters', type=int, default=0, help='timed iterations')
    p.add_argument('--duration', type=float, default=0, help='timed seconds (overrides --iters)')
    p.add_argument('--warmup', type=int, default=20)
    return p


def timed_loop(call, args, engine, device, threads, batch):
    for _ in range(args.warmup):
        call()
    rows = []
    t0 = time.perf_counter()
    i = 0
    while True:
        s = time.perf_counter()
        call()
        e = time.perf_counter()
        rows.append((args.test, engine, device, threads, batch, i,
                     f'{s - t0:.4f}', f'{(e - s) * 1000:.3f}'))
        i += 1
        if args.duration:
            if e - t0 >= args.duration:
                break
        elif i >= args.iters:
            break
    with open(args.out, 'a', newline='') as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(['test', 'engine', 'device', 'threads', 'batch',
                        'iter', 't_s', 'latency_ms'])
        w.writerows(rows)
    lat = sorted(r[-1] for r in rows)
    mean = sum(float(x) for x in lat) / len(lat)
    print(f'{args.test}: {len(rows)} calls, mean {mean:.2f} ms, '
          f'{1000 * batch / mean:.1f} inputs/s')
