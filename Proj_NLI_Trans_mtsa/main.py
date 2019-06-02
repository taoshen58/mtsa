import tensorflow as tf

from src.dataset import Dataset
from src.configs import ParamsCenter, Configs
from src.utils.file import load_file, save_file
from src.graph_handler import GraphHandler
from src.evaluator import Evaluator
from src.performance_recoder import PerformanceRecoder
import logging as log
import math


class MovingAverage(object):
    def __init__(self, decay=0.99):
        self.decay = decay
        self.value = None

    def __call__(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.decay * self.value + (1. - self.decay) * new_val
        return self.value

def train(cfg):
    # ======= data ===========
    loaded_data = load_file(cfg['processed_path'], 'processed_datasets', mode='pickle')
    if loaded_data is None and cfg['load_preproc']:
        train_data_obj = Dataset(cfg['train_data_path'], 'train', bpe_path=cfg['bpe_data_dir'])
        dev_data_obj = Dataset(cfg['dev_data_path'], 'dev', train_data_obj.bpe_enc)
        test_data_obj = Dataset(cfg['test_data_path'], 'test', train_data_obj.bpe_enc)
        dev_data_obj.sort_example()
        test_data_obj.sort_example()
        save_file({'train_data_obj': train_data_obj, 'dev_data_obj': dev_data_obj, 'test_data_obj': test_data_obj},
                  cfg['processed_path'])
    else:
        train_data_obj = loaded_data['train_data_obj']
        dev_data_obj = loaded_data['dev_data_obj']
        test_data_obj = loaded_data['test_data_obj']

    # 1. delete too long sentence, max len: 50
    train_data_obj.filter_example(cfg['max_sent_len'])

    # ========= build network ======
    if cfg['model_class'] is None:
        print('Did not find the model, please check '
              '1) module name 2) class name and 3) implementation of get_default_model_paramters(). exit!')
        exit()

    with tf.variable_scope("model") as scp:
        model = cfg['model_class'](
            cfg, train_data_obj.bpe_enc.n_vocab, 3,
            train_data_obj.bpe_enc.get_idx_from_token(Dataset.CLASSIFY_TOKEN),
            scp.name
        )

    # ======= build session =======
    graph_handler = GraphHandler(model, cfg)
    evaluator = Evaluator(model, cfg)
    performance_record = PerformanceRecoder(cfg['ckpt_dir'], cfg['save_model'], cfg['save_num'])

    sess = graph_handler.initialize()
    model.load_openai_pretrained_transformer_model(
        sess, cfg['pretrained_transformer_dir'])

    # ======== begin to train =======
    loss_task_ma, loss_lm_ma, accuracy_ma = MovingAverage(), MovingAverage(), MovingAverage()
    for example_batch, batch_num, data_round, idx_b in train_data_obj.generate_batch_iter(
            cfg['train_batch_size'], cfg['n_steps']):
        global_step_val = sess.run(model.global_step) + 1
        step_out = model.step(sess, example_batch, cfg['summary_period'])
        loss_task_ma(step_out['task_loss']), accuracy_ma(step_out['accuracy']), loss_lm_ma(step_out['lm_loss'])
        graph_handler.add_summary(step_out['summary'], global_step_val)
        if global_step_val % 100 == 0:
            log.info('data round: %d: %d/%d, global step:%d -- loss: %.4f, accu: %.4f' %
                     (data_round, idx_b, batch_num, global_step_val,
                      loss_task_ma.value, accuracy_ma.value))
            if 'lm_loss' in step_out:
                log.info('\tauxiliary language model perplexity: %.4f' % math.exp(loss_lm_ma.value))

        # eval
        if global_step_val % cfg['eval_period'] == 0:
            dev_res = evaluator.get_evaluation(sess, dev_data_obj, global_step_val)
            log.info('==> for dev, loss: %.4f, accuracy: %.4f' %
                     (dev_res['loss'], dev_res['accuracy']))
            criterion_metric = dev_res['accuracy']

            if not test_data_obj.no_label:
                test_res = evaluator.get_evaluation(sess, test_data_obj, global_step_val)
                log.info('~~> for test, loss: %.4f, accuracy: %.4f' %
                         (test_res['loss'], test_res['accuracy']))

            is_in_top, deleted_step = performance_record.update_top_list(global_step_val, criterion_metric, sess)
            if is_in_top:  # get prediction for non-labeled test data
                if test_data_obj.no_label and global_step_val > 0.4 * cfg['n_steps']:
                    evaluator.get_evaluation(sess, test_data_obj, global_step_val, write_predict=True)

            # todo: time count

    log.info(str(performance_record.top_list))




def main(_):
    param_center = ParamsCenter()
    cfg = Configs(param_center)

    if cfg['mode'] == 'train':
        train(cfg)



if __name__ == '__main__':
    tf.app.run()








