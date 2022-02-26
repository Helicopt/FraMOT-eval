# MULTI FRAME RATE EVALUATION

## Requirements

We need python >= 3.6 (conda new environment recommended), and install the following packages

- argparse
- numpy
- pillow
- senseToolkit
- trackeval

by excuting the followings:

```sh
python -m pip install argparse numpy pillow
cd senseToolkit
python setup.py build_ext --inplace install
cd ..
cd TrackEval
python -m pip install -r minimum_requirements.txt
python setup.py build install
cd ..
```


## Generate Dataset for Multi-frame-rate Evaluation

Run make_multi_frame_rate_dataset.py

```sh
usage: make_multi_frame_rate_dataset.py [-h] [--src SRC] [--dst DST] [--half]

optional arguments:
  -h, --help  show this help message and exit
  --src SRC   the source dataset dir, where sequence dirs are located
  --dst DST   the destination dataset dir, where the multi-frame-rate versions will be placed
  --half      split the half dataset
```

For example,
```sh
python make_multi_frame_rate_dataset.py --src /data/MOT20/test --dst /data/MOT20/mfr_test
```

## Evaluation on Results

Run eval_multi_frame_rates.py
```sh
usage: eval_multi_frame_rates.py [-h] [--gt-dir GTDIR] [--tracker-dir TRACKERDIR] [--logfile LOGFILE] [--match MATCH] [--trackers [TRACKERS [TRACKERS ...]]]

optional arguments:
  -h, --help            show this help message and exit
  --gt-dir GTDIR        the dir containing sequence dirs
  --tracker-dir TRACKERDIR
                        the dir containing results to eval
  --logfile LOGFILE     save the result metrics to file
  --match MATCH         regex to match sequences for eval
  --trackers [TRACKERS [TRACKERS ...]]
                        trackers to eval
```
Assume that all results to be evaluated are placed under /data/trackers, then
```sh
python eval_multi_frame_rates.py --gt-dir /data/MOT20/mfr_test --tracker-dir /data/trackers --logfile mot20_mfr_eval_results.json
```
Will automatically evaluate all tracker results under /data/trackers and generate the metrics in mot20_mfr_eval_results.json


