import math

import numpy as np
import tensorflow as tf

from configs import cfg
from snli_log_analysis import do_analyse_snli
from src.dataset import Dataset
from src.evaluator import Evaluator
from src.graph_handler import GraphHandler
from src.perform_recorder import PerformRecoder
from src.utils.file import load_file, save_file
from src.utils.record_log import _logger
from src.utils.time_counter import TimeCounter

# choose model
network_type = cfg.network_type
if network_type == 'mtsa':
    from src.model.m_mtsa import ModelMTSA as Model


def train():
    output_model_params()
    loadFile = True
    ifLoad, data = False, None
    if loadFile:
        ifLoad, data = load_file(cfg.processed_path, 'processed data', 'pickle')
    if not ifLoad or not loadFile:
        train_data_obj = Dataset(cfg.train_data_path, 'train')
        dev_data_obj = Dataset(cfg.dev_data_path, 'dev', dicts=train_data_obj.dicts)
        test_data_obj = Dataset(cfg.test_data_path, 'test', dicts=train_data_obj.dicts)

        save_file({'train_data_obj': train_data_obj, 'dev_data_obj': dev_data_obj, 'test_data_obj': test_data_obj},
                  cfg.processed_path)

        train_data_obj.save_dict(cfg.dict_path)
    else:
        train_data_obj = data['train_data_obj']
        dev_data_obj = data['dev_data_obj']
        test_data_obj = data['test_data_obj']

    train_data_obj.filter_data()
    dev_data_obj.filter_data()
    test_data_obj.filter_data()

    emb_mat_token, emb_mat_glove = train_data_obj.emb_mat_token, train_data_obj.emb_mat_glove

    with tf.variable_scope(network_type) as scope:
        model = Model(emb_mat_token, emb_mat_glove, len(train_data_obj.dicts['token']),
                      len(train_data_obj.dicts['char']), train_data_obj.max_lens['token'], scope.name)

    graphHandler = GraphHandler(model)
    evaluator = Evaluator(model)
    performRecoder = PerformRecoder(5)

    sess = tf.Session(config=sess_config_gene(cfg.gpu_mem))

    graphHandler.initialize(sess)

    # begin training
    steps_per_epoch = int(math.ceil(1.0 * train_data_obj.sample_num / cfg.train_batch_size))
    num_steps = cfg.num_steps or steps_per_epoch * cfg.max_epoch

    global_step = 0
    for sample_batch, batch_num, data_round, idx_b in train_data_obj.generate_batch_sample_iter(num_steps):
        global_step = sess.run(model.global_step) + 1
        if_get_summary = global_step % (cfg.log_period or steps_per_epoch) == 0
        loss, summary, train_op = model.step(sess, sample_batch, get_summary=if_get_summary)
        if global_step% 100 == 0:
            _logger.add('data round: %d: %d/%d, global step:%d -- loss: %.4f' %
                        (data_round, idx_b, batch_num, global_step, loss))

        if if_get_summary:
            graphHandler.add_summary(summary, global_step)

        # # occasional saving
        # if global_step % (cfg.save_period or steps_per_epoch) == 0:
        #     graphHandler.save(sess, global_step)

        # Occasional evaluation
        if (global_step > (cfg.num_steps - 300000) or cfg.model_dir_prefix=='test') and \
                                global_step % (cfg.eval_period or steps_per_epoch) == 0:
            # ---- dev ----
            dev_loss, dev_accu = evaluator.get_evaluation(
                sess, dev_data_obj, global_step
            )
            _logger.add('==> for dev, loss: %.4f, accuracy: %.4f' %
                        (dev_loss, dev_accu))
            # ---- test ----
            test_loss, test_accu = evaluator.get_evaluation(
                sess, test_data_obj, global_step
            )
            _logger.add('~~> for test, loss: %.4f, accuracy: %.4f' %
                        (test_loss, test_accu))
            is_in_top, deleted_step = performRecoder.update_top_list(global_step, dev_accu, sess)

        this_epoch_time, mean_epoch_time = cfg.time_counter.update_data_round(data_round)
        if this_epoch_time is not None and mean_epoch_time is not None:
            _logger.add('##> this epoch time: %f, mean epoch time: %f' % (this_epoch_time, mean_epoch_time))

            # if global_step % (cfg.save_period or steps_per_epoch) != 0:
            #     graphHandler.save(sess, global_step)
    _logger.writeToFile()
    do_analyse_snli(_logger.path)


def sess_config_gene(gpu_mem):
    if gpu_mem is None:
        gpu_options = tf.GPUOptions(allow_growth=True)
        graph_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    elif cfg.gpu_mem < 1.:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem)
        graph_config = tf.ConfigProto(gpu_options=gpu_options)
    else:
        gpu_options = tf.GPUOptions()
        graph_config = tf.ConfigProto(gpu_options=gpu_options)
    return graph_config


def test():
    assert cfg.load_path is not None
    output_model_params()
    loadFile = True
    ifLoad, data = False, None
    if loadFile:
        ifLoad, data = load_file(cfg.processed_path, 'processed data', 'pickle')
    if not ifLoad or not loadFile:
        raise RuntimeError('cannot find pre-processed dataset')
    else:
        train_data_obj = data['train_data_obj']
        dev_data_obj = data['dev_data_obj']
        test_data_obj = data['test_data_obj']

    train_data_obj.filter_data()
    dev_data_obj.filter_data()
    test_data_obj.filter_data()

    emb_mat_token, emb_mat_glove = train_data_obj.emb_mat_token, train_data_obj.emb_mat_glove

    with tf.variable_scope(network_type) as scope:
        model = Model(emb_mat_token, emb_mat_glove, len(train_data_obj.dicts['token']),
                      len(train_data_obj.dicts['char']), train_data_obj.max_lens['token'], scope.name)

    graphHandler = GraphHandler(model)
    evaluator = Evaluator(model)

    sess = tf.Session(config=sess_config_gene(cfg.gpu_mem))
    graphHandler.initialize(sess)

    # test model
    # ---- dev ----
    dev_loss, dev_accu = evaluator.get_evaluation(
        sess, dev_data_obj, None
    )
    _logger.add('==> for dev, loss: %.4f, accuracy: %.4f' %
                (dev_loss, dev_accu))
    # ---- test ----
    test_loss, test_accu = evaluator.get_evaluation(
        sess, test_data_obj, None
    )
    _logger.add('~~> for test, loss: %.4f, accuracy: %.4f' %
                (test_loss, test_accu))

    train_loss, train_accu = evaluator.get_evaluation(
        sess, train_data_obj, None
    )
    _logger.add('--> for test, loss: %.4f, accuracy: %.4f' %
                (train_loss, train_accu))


def inference_time():
    assert cfg.load_path is not None
    output_model_params()
    loadFile = True
    ifLoad, data = False, None
    if loadFile:
        ifLoad, data = load_file(cfg.processed_path, 'processed data', 'pickle')
    if not ifLoad or not loadFile:
        raise RuntimeError('cannot find pre-processed dataset')
    else:
        train_data_obj = data['train_data_obj']
        dev_data_obj = data['dev_data_obj']
        test_data_obj = data['test_data_obj']

    train_data_obj.filter_data()
    dev_data_obj.filter_data()
    test_data_obj.filter_data()

    emb_mat_token, emb_mat_glove = train_data_obj.emb_mat_token, train_data_obj.emb_mat_glove

    with tf.variable_scope(network_type) as scope:
        model = Model(emb_mat_token, emb_mat_glove, len(train_data_obj.dicts['token']),
                      len(train_data_obj.dicts['char']), train_data_obj.max_lens['token'], scope.name)

    graphHandler = GraphHandler(model)
    evaluator = Evaluator(model)

    sess = tf.Session(config=sess_config_gene(cfg.gpu_mem))
    graphHandler.initialize(sess)

    repeat_num = 10
    time_counter = TimeCounter()

    for i in range(5):
        evaluator.get_evaluation(sess, dev_data_obj, None)

    for t in range(repeat_num):
        dev_loss, dev_accu = evaluator.get_evaluation(
            sess, dev_data_obj, None, time_counter=time_counter
        )
        # _logger.add('~~> for test, loss: %.4f, accuracy: %.4f' %
        #             (test_loss, test_accu))
        time_rec = time_counter.update_data_round(t + 1)
        print('this: ', time_rec[0], '; all: ', time_rec[1])


def main(_):
    if cfg.mode == 'train':
        train()
    elif cfg.mode == 'test':
        test()
    elif cfg.mode == 'inference_time':
        inference_time()
    else:
        raise RuntimeError('no running mode named as %s' % cfg.mode)


def output_model_params():
    _logger.add()
    _logger.add('==>model_title: ' + cfg.model_name[1:])
    _logger.add()
    for key,value in cfg.args.__dict__.items():
        if key not in ['test','shuffle']:
            _logger.add('%s: %s' % (key, value))


if __name__ == '__main__':
    tf.app.run()



