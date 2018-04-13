from __future__ import absolute_import  
from __future__ import division  
from __future__ import print_function  
  
import tempfile  
from six.moves import urllib  
  
import pandas as pd  
import tensorflow as tf  
  
flags = tf.app.flags  
FLAGS = flags.FLAGS  
  
# 用来存放模型输出的目录设置，在第二个变量设置  
flags.DEFINE_string("model_dir", "model_dir", "Base directory for output models.")
# 用来设置用哪个模型来进行训练，在第二个变量设置，可选有：wide，deep，wide_n_deep  
flags.DEFINE_string("model_type", "deep",  
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")  
# 设置训练的步数，这里设置为200  
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")  
# 设置存放train_data的目录，在第二个变量设置  
flags.DEFINE_string(  
    "train_data",  
    "adult.data",
    "Path to the training data.")  
# 设置存放test_data的目录，在第二个变量设置  
flags.DEFINE_string(  
    "test_data",  
    "adult.test",
    "Path to the test data.")  
  
# 我们训练使用的数据的列的名称  
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",  
           "marital_status", "occupation", "relationship", "race", "gender",  
           "capital_gain", "capital_loss", "hours_per_week", "native_country",  
           "income_bracket"]  
LABEL_COLUMN = "label"  
  
""" 其实上面的数据的列可以分为两类，即categorical 和 continuous. 
categorical colum 就是这个列有有限个属性。 
例如workclass 有{ Private, Self-emp-not-inc, Self-emp-inc，etc} 
ccontinuous colum 就是这个列的属性是数字的连续型，如age 
"""  
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",  
                       "relationship", "race", "gender", "native_country"]  
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",  
                      "hours_per_week"]  
  
train_file_name='adult.data'
test_file_name ='adult.test'

def maybe_download():
    return train_file_name, test_file_name


def build_estimator(model_dir):
  """ 
  创建预测模型 
  """  
  # 创建稀疏的列. 列表中的每一个键将会获得一个从 0 开始的逐渐递增的id  
  # 例如 下面这句female 为 0，male为1。这种情况是已经事先知道列集合中的元素  
  # 都有那些  
  gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",  
                                                     keys=["female", "male"])  

  education = tf.contrib.layers.sparse_column_with_hash_bucket(  
      "education", hash_bucket_size=1000)  
  relationship = tf.contrib.layers.sparse_column_with_hash_bucket(  
      "relationship", hash_bucket_size=100)  
  workclass = tf.contrib.layers.sparse_column_with_hash_bucket(  
      "workclass", hash_bucket_size=100)  
  occupation = tf.contrib.layers.sparse_column_with_hash_bucket(  
      "occupation", hash_bucket_size=1000)  
  native_country = tf.contrib.layers.sparse_column_with_hash_bucket(  
      "native_country", hash_bucket_size=1000)  
  
  # 为连续的列元素设置一个实值列  
  age = tf.contrib.layers.real_valued_column("age")  
  education_num = tf.contrib.layers.real_valued_column("education_num")  
  capital_gain = tf.contrib.layers.real_valued_column("capital_gain")  
  capital_loss = tf.contrib.layers.real_valued_column("capital_loss")  
  hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")  
  
  # 为了更好的学习规律，收入是与年龄阶段有关的，因此需要把连续的数值划分  
  # 成一段一段的区间来表示收入  
  age_buckets = tf.contrib.layers.bucketized_column(age,  
                                                    boundaries=[  
                                                        18, 25, 30, 35, 40, 45,  
                                                        50, 55, 60, 65  
                                                    ])  
  
  # 上面所说的模型，  
  # 这个为 wide 模型  
  wide_columns = [gender, native_country, education, occupation, workclass,  
                  relationship, age_buckets,  
                  tf.contrib.layers.crossed_column([education, occupation],  
                                                   hash_bucket_size=int(1e4)),  
                  tf.contrib.layers.crossed_column(  
                      [age_buckets, education, occupation],  
                      hash_bucket_size=int(1e6)),  
                  tf.contrib.layers.crossed_column([native_country, occupation],  
                                                   hash_bucket_size=int(1e4))]  
  
  # 这个为 deep 模型  
  deep_columns = [  
      tf.contrib.layers.embedding_column(workclass, dimension=8),  
      tf.contrib.layers.embedding_column(education, dimension=8),  
      tf.contrib.layers.embedding_column(gender, dimension=8),  
      tf.contrib.layers.embedding_column(relationship, dimension=8),  
      tf.contrib.layers.embedding_column(native_country,  
                                         dimension=8),  
      tf.contrib.layers.embedding_column(occupation, dimension=8),  
      age,  
      education_num,  
      capital_gain,  
      capital_loss,  
      hours_per_week,  
  ]  
  
  # 判断选的是以哪个模型来进行训练  
  # 返回模型  
  if FLAGS.model_type == "wide":  
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,  
                                          feature_columns=wide_columns)  
  elif FLAGS.model_type == "deep":  
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,  
                                       feature_columns=deep_columns,  
                                       hidden_units=[100, 50])  
  else:  
    m = tf.contrib.learn.DNNLinearCombinedClassifier(  
        model_dir=model_dir,  
        linear_feature_columns=wide_columns,  
        dnn_feature_columns=deep_columns,  
        dnn_hidden_units=[100, 50])  
  return m  
  
  
def input_fn(df):  
  """这个函数的主要作用就是把输入数据转换成tensor，即向量型"""  
    
  # 为continuous colum列的每一个属性创建一个对于的 dict 形式的 map  
  # 对应列的值存储在一个 constant 向量中  
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}  
  # 为 categorical colum列的每一个属性创建一个对于的 dict 形式的 map  
  # 对应列的值存储在一个 tf.SparseTensor 中  
  categorical_cols = {k: tf.SparseTensor(  
      indices=[[i, 0] for i in range(df[k].size)],  
      values=df[k].values,  
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}  
  # 合并上面两个dict类型  
  feature_cols = dict(continuous_cols)  
  feature_cols.update(categorical_cols)  
  
  # 将 label column 转换成一个 constant 向量  
  label = tf.constant(df[LABEL_COLUMN].values)  
    
  # 返回向量形式对应列的数据和label  
  return feature_cols, label  
  
  
def train_and_eval():  
  """这个函数是真正的入口函数，用来训练数据， 
    之后才进行 evaluate。 
  """  
  # 首先取得train 和 test 文件的文件名  
  train_file_name, test_file_name = maybe_download()  
  
  # 用 pandas 读入数据  
  df_train = pd.read_csv(  
      tf.gfile.Open(train_file_name),  
      names=COLUMNS,  
      skipinitialspace=True,  
      engine="python")  
  df_test = pd.read_csv(  
      tf.gfile.Open(test_file_name),  
      names=COLUMNS,  
      skipinitialspace=True,  
      skiprows=1,  
      engine="python")  
  
  # 移除非数字  
  df_train = df_train.dropna(how='any', axis=0)  
  df_test = df_test.dropna(how='any', axis=0)  
  
  # 将 收入一列 即label 转换为 0和1，即大于50K的设置为1  
  # 小于50K的设置为0  
  df_train[LABEL_COLUMN] = (  
      df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)  
  df_test[LABEL_COLUMN] = (  
      df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)  
  
  # 判断输出的目录是否存在，不存在则创建临时的  
  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir  
  print("model directory = %s" % model_dir)  
  
  # 创建预测模型，返回的是 wide 或者 deep 或者 wide&deep 模型中的一个  
  m = build_estimator(model_dir)  
  
  # 进行训练  
  m.fit(input_fn=lambda: input_fn(df_train), steps=FLAGS.train_steps)  
  
  # 使用test 数据进行评价  
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)  
  for key in sorted(results):  
    print("%s: %s" % (key, results[key]))  
  
  
def main(_):  
  train_and_eval()  
  
  
if __name__ == "__main__":  
  tf.app.run()  
