
from const.config_const import *
from const.constants import shared_incorrect_para_msg
from tests.run_testcase import test_loop


def test_config(config, logger):
    tests = [
        {
            'name': f'test {GENERAL_TXT.batch_size}',
            'result': int(config.batch_size) > 0,
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=GENERAL_TXT.batch_size) + ' It should be > 0'
        },

        {
            'name': f'test {ANALYZE_FREQ.r_values}',
            'result': all(r >= 0 and r <= 452 for r in config.r_values),
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=ANALYZE_FREQ.r_values) + ' All r_values should be >= 0 and <= 452'
        },

        {
            'name': f'test {MODEL_TXT.model_type}',
            'result': config.model_type in ["3", "5n", "5s", "5m", "5l", "5x", "6n", "6s", "6m", "6l", "6x",
                                            "8n", "8s", "8m", "8l", "8x"],
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=MODEL_TXT.model_type) + f' It should be one of these: [yolov5s]'
        },
        {
            'name': f'test {MODEL_TXT.score_threshold}',
            'result': config.score_threshold >= 0,
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=MODEL_TXT.score_threshold) + ' Threshold should be >= 0'
        },
    ]
    ok = test_loop(test_cases=tests, test_name='validate input config', logger=logger)
    return ok
