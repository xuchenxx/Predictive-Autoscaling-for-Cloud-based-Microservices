import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima_model import ARIMA

from json import JSONDecodeError
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
from dataclasses_json import dataclass_json, LetterCase


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
    print("No standard input provided to ARIMA algorithm, exiting", file=sys.stderr)
    sys.exit(1)

try:
    algorithm_input = AlgorithmInput.from_json(stdin)
except JSONDecodeError as ex:
    print("Invalid JSON provided: {0}, exiting".format(str(ex)), file=sys.stderr)
    sys.exit(1)
except KeyError as ex:
    print("Invalid JSON provided: missing {0}, exiting".format(str(ex)), file=sys.stderr)
    sys.exit(1)

stdin

try:
  x = 5
  count = 50
  data = {"Date": [10 * num for num in range(count)], "Replicas": [float(5 * num) for num in range(count)]}
  print(data)
  df = pd.DataFrame(data)

  # df.drop(df.head(2000).index, inplace=True)
  df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
  indexed_df = df.set_index(['Date'])
  indexed_df = indexed_df[indexed_df != 0]
  # indexed_df.fillna(method='ffill')

  indexed_df_logscale = np.log(indexed_df)
  indexed_df_logscale = indexed_df_logscale.dropna()
  indexed_df_logscale_values = [d for d in indexed_df_logscale['Replicas'].values]
  initial_log_val = indexed_df_logscale.values[0]
  indexed_df_diff_logscale = indexed_df_logscale - indexed_df_logscale.shift()
  indexed_df_diff_logscale = indexed_df_diff_logscale.dropna()
  indexed_df_diff_logscale_values = [d[0] for d in indexed_df_diff_logscale.values]

  diff_logscale_predictions = []
  test_set = indexed_df_diff_logscale_values[-x:]
  history = indexed_df_diff_logscale_values[:-x]
  last_training_set_val = history[-1]

  for i, true_val in enumerate(test_set):
    print("Predicting " + str(i))

    # Fit the model
    model = ARIMA(history, order=(0,1,0))
    fitted_model = model.fit(disp=-1)

    # Predicting the next value
    diff_logscale_prediction = fitted_model.forecast()[0]
    # diff_logscale_prediction = [true_val]

    # Update the predictions with the new estimate
    # Update the history with the real value
    diff_logscale_predictions.append(diff_logscale_prediction[0])
    history.append(true_val)

  assert len(test_set) == len(diff_logscale_predictions)

  logscale_predictions = []
  logscale_history = list(np.cumsum(np.concatenate((np.array(initial_log_val), indexed_df_diff_logscale_values[:-x]))))
  for i, diff in enumerate(diff_logscale_predictions):
      logscale_predictions.append(logscale_history[-1] + diff)
      logscale_history.append(logscale_history[-1] + test_set[i])

  predictions = np.exp(logscale_predictions)
  print(int(predictions[-1]), end="")
except Exception as err:
  print(err)
