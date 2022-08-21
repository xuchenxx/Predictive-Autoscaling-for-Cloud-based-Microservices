from json import JSONEncoder, JSONDecodeError
import json
import sys
import os
import pickle
import numpy
import pandas as pd
from pmdarima.arima import auto_arima


MODEL_LOCATION = '/tmp/arima.pkl'
ACTION='action'
INDEX='created'
TRAIN='train'
PREDICT='predict'
LOOKAHEAD='lookAhead'


class NumpyArrayEncoder(JSONEncoder):
    '''
    NumpyArraryEncoder class
    '''

    def default(self, o):
        '''
        obj: ndarray to json
        '''
        if isinstance(o, numpy.ndarray):
            return o.tolist()
        return JSONEncoder.default(self, o)


def train(request_json):
    '''
    Post method to return default json object with 200OK
    '''
    data = pd.DataFrame.from_dict(request_json, orient="index"
                                  ).T.set_index("Date")
    stepwise_model = auto_arima(
        data,
        start_p=1,
        start_q=1,
        max_p=3,
        max_q=3,
        m=12,
        start_P=0,
        seasonal=True,
        d=1,
        D=1,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        )
    # print(stepwise_model.aic())
    # print(stepwise_model.summary())

    # Serialize with Pickle

    with open('/tmp/arima.pkl', 'wb') as pkl:
        pickle.dump(stepwise_model, pkl)

    return json.dumps('{"success":1}')


def load_pred(location):
    '''
    Load prediction from location
    location: default path to load prediction from
    Returns: Picket object
    '''
    f_d = open(location, 'rb')
    return pickle.load(f_d)


def update_model(location, data):
    '''
    Update model with latest data
    location: Location of model to load from
    data: new data to update the model with
    Returns: pickled model
    '''
    model = load_pred(location)
    model.update(data)
    with open(location, 'wb') as pkl:
        pickle.dump(model, pkl)
    return model


def predict(request_json,n_periods):
    '''
    Load prediction and update them with existing model
    Returns: Jsonify data
    '''
    data = pd.DataFrame.from_dict(request_json, orient="index"
                                  ).T.set_index("Date")
    model = update_model(MODEL_LOCATION, data)
    (prediction, new_conf_int) = model.predict(n_periods=n_periods, return_conf_int=True)
    # print(new_conf_int)
    return json.dumps(prediction, cls=NumpyArrayEncoder)

if __name__ == '__main__':
    stdin = sys.stdin.read()
    if stdin is not None:
        try:
            input_json = json.loads(stdin)
            print(input_json)
            data = {"Date": [10 * num for num in range(12)], "Replicas": [5] * 12}
            train(data)
            predict(data, input_json.get(LOOKAHEAD,3))

        except JSONDecodeError as ex:
            print("Invalid JSON provided: {0}, exiting".format(str(ex)), file=sys.stderr)
            sys.exit(1)
        except Exception as ex:
            print("Exception while processing SARIMA algorithm", ex)
            sys.exit(1)
    else:
        print("No standard input provided to SARIMA algorithm, exiting", file=sys.stderr)
        sys.exit(1)
