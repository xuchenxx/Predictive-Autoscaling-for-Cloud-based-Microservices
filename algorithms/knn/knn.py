# Copyright 2020 The Predictive Horizontal Pod Autoscaler Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=no-member, invalid-name

"""
This linear regression script performs a linear regression using the provided values and configuration using the
statsmodel library.
"""

import sys
import math
import numpy as np
from json import JSONDecodeError
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
from dataclasses_json import dataclass_json, LetterCase

# Takes in list of stored evaluations and the look ahead value:
# {
#   "lookAhead": 3,
#   "evaluations": [
#       {
#           "id": 0,
#           "created": "2020-02-01T00:55:33Z",
#           "val": {
#               "targetReplicas": 3
#           }
#       },
#       {
#           "id": 1,
#           "created": "2020-02-01T00:56:33Z",
#           "val": {
#               "targetReplicas": 2
#           }
#       }
#   ]
# }

@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class EvaluationValue:
    """
    JSON data representation of an evaluation value, contains the scaling target replicas
    """
    target_replicas: int

@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class Evaluation:
    """
    JSON data representation of a timestamped evaluation
    """
    id: int
    created: str
    val: EvaluationValue

@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class AlgorithmInput:
    """
    JSON data representation of the data this algorithm requires to be provided to it.
    """
    look_ahead: int
    evaluations: List[Evaluation]
    current_time: Optional[str] = None

stdin = sys.stdin.read()

if stdin is None or stdin == "":
    print("No standard input provided to KNN algorithm, exiting", file=sys.stderr)
    sys.exit(1)

try:
    algorithm_input = AlgorithmInput.from_json(stdin)
except JSONDecodeError as ex:
    print("Invalid JSON provided: {0}, exiting".format(str(ex)), file=sys.stderr)
    sys.exit(1)
except KeyError as ex:
    print("Invalid JSON provided: missing {0}, exiting".format(str(ex)), file=sys.stderr)
    sys.exit(1)


def predict(training_data, k):
    prediction = sum(training_data[-k-1:-1])/k
    return math.floor(prediction * 100) / 100.0

try:
    x = []
    y = []

    training_data = []
    scaled_predictions = []
    accuracies = []
    k = 5

    evaluation_data = np.array(list(map(lambda x: x.val.target_replicas, algorithm_input.evaluations)))
    scaled_data = evaluation_data - np.concatenate((np.array([0]), evaluation_data[:-1]))

    for i, replicas in enumerate(scaled_data):
        training_data.append(float(replicas))

        prediction = predict(training_data, k)
        scaled_predictions.append(prediction)

    # Throw out the first k predictions since they peeked at actual data.
    for i in range(k):
        scaled_predictions[i] = 0.0

    # Transform back to original scale
    predictions = []
    scaled_predictions[k-1] = evaluation_data[k-1]

    for i in range(len(evaluation_data)):
        if i < k:
            predictions.append(0)
        else:
            predictions.append(evaluation_data[i-1] + scaled_predictions[i])

    print(int(predictions[-1]), end="")
except Exception as e:
    print(e)
