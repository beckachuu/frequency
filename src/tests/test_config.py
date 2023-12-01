
from const.config_const import FREQUENCY_TXT, GENERAL_TXT, MODEL_TXT
from const.constants import shared_incorrect_para_msg
from tests.run_testcase import test_loop


def test_config(config, logger):
    tests = [
        {
            'name': f'test {GENERAL_TXT.demo_count}',
            'result': int(config.demo_count) >= 0,
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=GENERAL_TXT.demo_count) + ' It should be >= 0'
        },
        {
            'name': f'test {GENERAL_TXT.batch_size}',
            'result': int(config.batch_size) > 0,
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=GENERAL_TXT.batch_size) + ' It should be > 0'
        },

        {
            'name': f'test {FREQUENCY_TXT.r_values}',
            'result': all(int(r) > 0 for r in config.r_values),
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=FREQUENCY_TXT.r_values) + ' All r_values should be > 0'
        },

        {
            'name': f'test {MODEL_TXT.model_type}',
            'result': config.model_type in ["yolov5s"],
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=MODEL_TXT.model_type) + f' It should be one of these: [yolov5s]'
        },
        {
            'name': f'test {MODEL_TXT.score_thresholds}',
            'result': all(thres >= 0 for thres in config.score_thresholds),
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=MODEL_TXT.score_thresholds) + ' All thresholds should be >= 0'
        },
    ]
    ok = test_loop(test_cases=tests, test_name='validate input config', logger=logger)
    return ok
