"""BirdNET V2.4 FP32 TFLite on CPU: the workload BirdNET-Go runs.

One call = one 3 s, 48 kHz window (144000 samples). Real-time factor is
3 s / latency, computed at collation time. Random noise is used as input:
inference cost does not depend on the audio content.
"""
import argparse
import os
import numpy as np
from ai_edge_litert.interpreter import Interpreter

from common import add_args, timed_loop

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                     'models', 'BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite')

p = add_args(argparse.ArgumentParser())
p.add_argument('--threads', type=int, required=True)
args = p.parse_args()

interp = Interpreter(model_path=MODEL, num_threads=args.threads)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
audio = np.random.uniform(-1, 1, inp['shape']).astype(inp['dtype'])


def call():
    interp.set_tensor(inp['index'], audio)
    interp.invoke()


timed_loop(call, args, 'birdnet-tflite-fp32', 'cpu', args.threads, 1)
