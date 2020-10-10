#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys
from pprint import pprint

args = sys.argv

if len(args) < 1:
  sys.exit()

model_path = args[1]

interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
tensor_details = interpreter.get_tensor_details()
ops_details = interpreter._get_ops_details()

pprint(input_details)
pprint(tensor_details[0])
pprint(tensor_details[10])

pprint(output_details)
#pprint(len(tensor_details))
#pprint(len(ops_details))

#pprint(ops_details)

#pprint(ops_details[1])
#pprint(tensor_details[0])

