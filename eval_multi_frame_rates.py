import trackeval
import argparse
import os
import tempfile
import glob
import re
# import torch
import numpy as np
import json
import time
import uuid

skips = [1, 2, 4, 8, 16, 25, 36, 50, 75]
GT_ROOT = '/mnt/lustre/share/fengweitao/MOT20/mfr_train/'
TRACKER_ROOT = '/mnt/lustre/share/fengweitao/MOT20/trackers/'


def to_scaler(data):
    if isinstance(data, (tuple, list, set)):
        return [to_scaler(x) for x in data]
    if isinstance(data, dict):
        return {k: to_scaler(v) for k, v in data.items()}
    # if isinstance(data, torch.Tensor):
    #     return to_scaler(data.detach().cpu().numpy())
    if isinstance(data, np.ndarray):
        return np.mean(data)
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    return data


class trackevalator(object):

    def __init__(self, gt_dir, tracker_dir, match=None):
        self.gt_path = gt_dir
        self.track_eval_cfg = dict(
            METRICS=["HOTA", "CLEAR", "Identity"],
            # METRICS: ["HOTA", "CLEAR"]
            THRESHOLD=0.5,
            BENCHMARK='MOT17',
            USE_PARALLEL=True,
            NUM_PARALLEL_CORES=8
        )
        self.fast_eval = True
        self.group_by = ['S-%d-' % s for s in skips]
        self.formatter = "{root}/{seq}/{fr}.{ext}"
        self.tracker_path = tracker_dir
        self.tracker_names = list(filter(lambda x: os.path.isdir(
            os.path.join(tracker_dir, x)), os.listdir(tracker_dir)))
        self.pattern = match
        self.build_trackeval_evaluator(self.track_eval_cfg)

    def build_trackeval_evaluator(self, _cfg):
        def update_value(_cfg, out_cfg):
            for key, val in _cfg.items():
                key = key.upper()
                if key in out_cfg.keys():
                    out_cfg[key] = val
            return out_cfg

        self.trackeval_eval_config = update_value(
            _cfg, self.get_default_trackeval_config())
        self.trackeval_eval_config['DISPLAY_LESS_PROGRESS'] = False
        self.trackeval_dataset_config = update_value(
            _cfg, self.get_default_track_eval_dataset_config())
        self.trackeval_metric_config = {
            key: value for key, value in _cfg.items() if key in ['METRICS', 'THRESHOLD']}
        self.trackeval_config = {**self.trackeval_eval_config,
                                 ** self.trackeval_dataset_config, **self.trackeval_metric_config}

        self.trackeval_evaluator = trackeval.Evaluator(
            self.trackeval_eval_config)
        self.trackeval_metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
            if metric.get_name() in self.trackeval_metric_config['METRICS']:
                self.trackeval_metrics_list.append(
                    metric(self.trackeval_metric_config))
        if len(self.trackeval_metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')

    @staticmethod
    def get_default_trackeval_config():
        """Returns the default config values for evaluation"""
        code_path = os.path.abspath('.')
        default_config = {
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 8,
            'BREAK_ON_ERROR': True,  # Raises exception and exits with error
            'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
            # if not None, save any errors into a log file.
            'LOG_ON_ERROR': os.path.join(code_path, 'error_log.txt'),

            'PRINT_RESULTS': True,
            'PRINT_ONLY_COMBINED': True,
            'PRINT_CONFIG': True,
            'TIME_PROGRESS': True,
            'DISPLAY_LESS_PROGRESS': True,

            'OUTPUT_SUMMARY': False,
            # If False, summary files are not output for classes with no detections
            'OUTPUT_EMPTY_CLASSES': True,
            'OUTPUT_DETAILED': True,
            'PLOT_CURVES': True,
        }
        return default_config

    def get_default_track_eval_dataset_config(self):
        """Default class config values"""
        default_config = {
            # Location of GT data
            'GT_FOLDER': self.gt_path,
            # Trackers location
            'TRACKERS_FOLDER': self.tracker_path,
            # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'OUTPUT_FOLDER': None,
            # Filenames of trackers to eval (if None, all in folder)
            'TRACKERS_TO_EVAL': self.tracker_names,
            'CLASSES_TO_EVAL': ['pedestrian'],  # Valid: ['pedestrian']
            'BENCHMARK': 'MOT17',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
            'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            # Whether to perform preprocessing (never done for MOT15)
            'DO_PREPROC': True,
            # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'TRACKER_SUB_FOLDER': '',
            # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',
            # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'TRACKER_DISPLAY_NAMES': None,
            # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
            'SEQMAP_FOLDER': None,
            # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            'SEQMAP_FILE': None,
            # If not None, directly specify sequences to eval and their number of timesteps
            'SEQ_INFO': None,
            # '{gt_folder}/{seq}/gt/gt.txt'
            'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
            'SKIP_SPLIT_FOL': True,  # If False, data is in GT_FOLDER/BENCHMARK-SPLIT_TO_EVAL/ and in
            # TRACKERS_FOLDER/BENCHMARK-SPLIT_TO_EVAL/tracker/
            # If True, then the middle 'benchmark-split' folder is skipped for both.
        }
        return default_config

    def get_track_prec(self, logfile=None, trackers=None):

        seq_name_list = sorted(os.listdir(self.gt_path))
        if self.pattern is not None:
            seq_name_list = list(filter(lambda x: re.findall(self.pattern, x), seq_name_list))
        # seq_name_list = list(map(lambda x: x[:-4],
        #                          filter(lambda x: x.endswith('.txt'), seq_name_list)))

        # process dt
        if trackers is None:
            trackers = self.tracker_names
        print('tracker_name: {}'.format(trackers))

        self.trackeval_dataset_config['TRACKERS_TO_EVAL'] = trackers

        if self.group_by is None or not isinstance(self.group_by, (list, tuple)) or not self.group_by:

            fd = tempfile.NamedTemporaryFile('w')
            seq_list_path = fd.name
            fd.write('name\n')
            for seq in seq_name_list:
                fd.write(seq + '\n')
            fd.flush()

            self.trackeval_dataset_config['SEQMAP_FILE'] = seq_list_path
            track_eval_dataset_list = [
                trackeval.datasets.MotChallenge2DBox(self.trackeval_dataset_config)]

            output_res, output_msg = self.trackeval_evaluator.evaluate(
                track_eval_dataset_list, self.trackeval_metrics_list)

            metrics = {tracker_name: output_res['MotChallenge2DBox'][tracker_name]
                       ['COMBINED_SEQ']['pedestrian'] for tracker_name in trackers}

            fd.close()
        else:
            metrics = {}
            for i, rep in enumerate(self.group_by):
                sub_list = []
                for seq_name_i in seq_name_list:
                    if seq_name_i.startswith(rep):
                        sub_list.append(seq_name_i)
                if sub_list:
                    fd = tempfile.NamedTemporaryFile('w')
                    seq_list_path = fd.name
                    fd.write('name\n')
                    for seq in sub_list:
                        fd.write(seq + '\n')
                    fd.flush()
                else:
                    continue
                self.trackeval_dataset_config['SEQMAP_FILE'] = seq_list_path
                track_eval_dataset_list = [
                    trackeval.datasets.MotChallenge2DBox(self.trackeval_dataset_config)]

                output_res, output_msg = self.trackeval_evaluator.evaluate(
                    track_eval_dataset_list, self.trackeval_metrics_list)

                res = {tracker_name: output_res['MotChallenge2DBox'][tracker_name]
                       ['COMBINED_SEQ']['pedestrian'] for tracker_name in trackers}
                metrics[rep] = res
                fd.close()
        metrics = to_scaler(metrics)
        if logfile is None or os.path.isdir(logfile):
            root = logfile
            if logfile is None:
                root = '.'
            stamp = time.strftime('%Y-%m-%d-%H-%M-%S')
            uid = uuid.uuid4()
            logfile = os.path.join(os.path.abspath(root), 'metrics-%s-%s.json' % (stamp, uid))
        with open(logfile, 'w') as fd:
            json.dump(metrics, fd)
        return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-dir', dest='gtdir', type=str, default=GT_ROOT, help='the dir containing sequence dirs')
    parser.add_argument('--tracker-dir', dest='trackerdir', type=str,
                        default=TRACKER_ROOT, help='the dir containing results to eval')
    parser.add_argument('--logfile', type=str, default=None, help='save the result metrics to file')
    parser.add_argument('--match', type=str, default=None, help='regex to match sequences for eval')
    parser.add_argument('--trackers', nargs='*', default=None, help='trackers to eval')

    args = parser.parse_args()
    handler = trackevalator(gt_dir=args.gtdir, tracker_dir=args.trackerdir, match=args.match)
    metrics = handler.get_track_prec(logfile=args.logfile, trackers=args.trackers)
    # print(metrics)
