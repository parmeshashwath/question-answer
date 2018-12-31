from flask import Flask, jsonify, request, url_for, render_template
import logging
from qa_model import QAModel
from vocab import get_glove
from official_eval_helper import get_json, generate_answers, generate_answers_prob
from main import FLAGS
import tensorflow as tf
import os
import json
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir


# Some GPU settings
config=tf.ConfigProto()
config.gpu_options.allow_growth = True
print "glove_path"
FLAGS.glove_path = FLAGS.glove_path or os.path.join(DEFAULT_DATA_DIR, "glove.6B.{}d.txt".format(FLAGS.embedding_size))

print FLAGS.glove_path
emb_matrix, word2id, id2word = get_glove(FLAGS.glove_path, FLAGS.embedding_size)
qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix)


def initialize_model(session, model, train_dir, expect_exists):
    """
    Initializes model from train_dir.

    Inputs:
      session: TensorFlow session
      model: QAModel
      train_dir: path to directory where we'll look for checkpoint
      expect_exists: If True, throw an error if no checkpoint is found.
        If False, initialize fresh model if no checkpoint is found.
    """
    print "Looking for model at %s..." % train_dir
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print "Reading model parameters from %s" % ckpt.model_checkpoint_path
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at %s" % train_dir)
        else:
            print "There is no saved checkpoint at %s. Creating model with fresh parameters." % train_dir
            session.run(tf.global_variables_initializer())
            print 'Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables())


# Load best model
sess =  tf.Session(config=config)

# Load model from ckpt_load_dir
FLAGS.ckpt_load_dir = 'experiments/bidaf_best'
initialize_model(sess, qa_model, FLAGS.ckpt_load_dir, expect_exists=True)


@app.route('/')
def static_page():
    return render_template('index.html')

@app.route('/getAnswers',methods=['POST'])
def getAnswers():
    data = json.loads(request.data)
    print data
    qn_uuid_data, context_token_data, qn_token_data = get_json(data)
    print "qn_uuid_data"
    print qn_uuid_data
    print "#"*100
    print "context_token_data"
    print context_token_data
    print "#"*100
    print "qn_token_data"
    print qn_token_data
    answers_dict = generate_answers(sess, qa_model, word2id, qn_uuid_data, context_token_data, qn_token_data)
    return jsonify(answers_dict)



@app.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port="5556",debug=True)
