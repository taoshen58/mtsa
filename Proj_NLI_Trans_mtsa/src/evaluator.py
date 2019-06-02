import numpy as np
import tensorflow as tf
import logging as log
import csv
from src.utils.file import mp_join
from os.path import join

class Evaluator(object):
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

        self.global_step = model.global_step
        ## ---- summary----
        self.build_summary()
        self.writer = tf.summary.FileWriter(cfg['summary_dir'])

    def get_evaluation(self, sess, dataset_obj, global_step=None, time_counter=None, write_predict=False):
        if write_predict:
            self.get_prediction(sess, dataset_obj, global_step, time_counter)

        if not dataset_obj.no_label:
            return self.get_evaluation_with_label(sess, dataset_obj, global_step, time_counter)

        return None

    def get_prediction(self, sess, dataset_obj, global_step=None, time_counter=None):
        label_id2str = {
            0: "contradiction",
            1: "neutral",
            2: "entailment",
        }

        global_step = str(global_step)
        file_name = "{dataset}_{step}.csv".format(dataset=self.cfg['dataset'], step=global_step)
        log.info('getting evaluation result for %s' % dataset_obj.data_type)
        predict_dir = mp_join(self.cfg['other_dir'], "predict_{dataset}".format(dataset=self.cfg['dataset']))

        with open(join(predict_dir, file_name), 'w', encoding='utf-8') as fp:
            csv_writer = csv.writer(fp, delimiter=',')
            csv_writer.writerow(['pairID', 'gold_label'])

            for sample_batch, _, _, _ in dataset_obj.generate_batch_iter(self.cfg['test_batch_size']):
                feed_dict = self.model.get_feed_dict(sample_batch, False)
                if time_counter is not None:
                    time_counter.add_start()
                logits = sess.run(self.model.logits, feed_dict)
                if time_counter is not None:
                    time_counter.add_stop()
                predicts = np.argmax(logits, -1)

                for sample, predict in zip(sample_batch, predicts):
                    predict_label = label_id2str[int(predict)]
                    csv_writer.writerow([sample['pairID'], predict_label])

    def get_evaluation_with_label(self, sess, dataset_obj, global_step=None, time_counter=None):
        log.info('getting evaluation result for %s' % dataset_obj.data_type)
        logits_list, loss_list, accu_list = [], [], []
        for sample_batch, _, _, _ in dataset_obj.generate_batch_iter(self.cfg['test_batch_size']):
            feed_dict = self.model.get_feed_dict(sample_batch, False)
            if time_counter is not None:
                time_counter.add_start()
            logits, loss, accu = sess.run([self.model.logits,
                                           self.model.task_loss,
                                           self.model.accuracy],
                                          feed_dict)
            if time_counter is not None:
                time_counter.add_stop()
            logits_list.append(np.argmax(logits, -1))
            loss_list.append(loss)
            accu_list.append(accu)

        logits_array = np.concatenate(logits_list, 0)
        loss_value = np.mean(loss_list)
        accu_array = np.concatenate(accu_list, 0)
        accu_value = np.mean(accu_array)

        if global_step is not None:
            if dataset_obj.data_type == 'train':
                summary_feed_dict = {
                    self.train_loss: loss_value,
                    self.train_accuracy: accu_value,
                }
                summary = sess.run(self.train_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)
            elif dataset_obj.data_type == 'dev':
                summary_feed_dict = {
                    self.dev_loss: loss_value,
                    self.dev_accuracy: accu_value,
                }
                summary = sess.run(self.dev_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)
            else:
                summary_feed_dict = {
                    self.test_loss: loss_value,
                    self.test_accuracy: accu_value,
                }
                summary = sess.run(self.test_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)

        return {'loss': loss_value, 'accuracy': accu_value}


    # --- internal use ------
    def build_summary(self):
        with tf.name_scope('train_summaries'):
            self.train_loss = tf.placeholder(tf.float32, [], 'train_loss')
            self.train_accuracy = tf.placeholder(tf.float32, [], 'train_accuracy')
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_loss', self.train_loss))
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_accuracy', self.train_accuracy))
            self.train_summaries = tf.summary.merge_all('train_summaries_collection')

        with tf.name_scope('dev_summaries'):
            self.dev_loss = tf.placeholder(tf.float32, [], 'dev_loss')
            self.dev_accuracy = tf.placeholder(tf.float32, [], 'dev_accuracy')
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_loss',self.dev_loss))
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_accuracy',self.dev_accuracy))
            self.dev_summaries = tf.summary.merge_all('dev_summaries_collection')

        with tf.name_scope('test_summaries'):
            self.test_loss = tf.placeholder(tf.float32, [], 'test_loss')
            self.test_accuracy = tf.placeholder(tf.float32, [], 'test_accuracy')
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_loss',self.test_loss))
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_accuracy',self.test_accuracy))
            self.test_summaries = tf.summary.merge_all('test_summaries_collection')




