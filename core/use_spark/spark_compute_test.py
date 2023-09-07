import configparser
import logging
import os
import uuid

# todo 用spark集群处理高耗时高内存占用的部分，以下是示例代码

from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, StructType

logger = logging.getLogger('Log')


def xtb_default(sdf_data):
    index = uuid.uuid4()
    accuracy = 2

    prepare_data_cmd = f"mkdir -p /home/data/{index} && cd /home/data/{index}/ && rm -rf * && mkdir -p /home/data/{index}/work "
    res = os.system(prepare_data_cmd)
    if res != 0:
        logger.info(f"run cmd: {prepare_data_cmd} result:{res}")

    p = 1
    in_sdf = "input.sdf"
    input_sdf_file = f"/home/data/{index}/work/{in_sdf}"
    fo = open(input_sdf_file, "w")
    fo.write(sdf_data)
    # 刷新缓冲区
    fo.flush()
    fo.close()

    # 之前运行os.system提示 code 32512，需要指定xtb绝对路径
    cmd = f'cd /home/data/{index}/work && /xtb-6.4.0/bin/xtb {in_sdf} --gfn {accuracy} --opt -P {p}  1>/dev/null 2>/dev/null'
    res = os.system(cmd)
    if res != 0:
        logger.error(f"run cmd: {cmd} result:{res}")
        return ''

    result_file = f'/home/data/{index}/work/xtbopt.log'
    if os.path.exists(result_file):
        file_data = open(result_file, "r").read()
        return file_data
    else:
        logger.error(f"run cmd: {cmd} result:{res}")

    return ''


# 这个配置文件，会在启动任务时集群自动配置
config = configparser.ConfigParser()
config.read("/opt/spark/default-config/spark-defaults.conf")
print(config)

AWS_ACCESS_KEY_ID = config['aws']['spark.hadoop.fs.s3a.access.key']
AWS_SECRET_ACCESS_KEY = config['aws']['spark.hadoop.fs.s3a.secret.key']
AWS_ENDPOINT_ADDRESS = config['aws']['spark.hadoop.fs.s3a.endpoint']
AWS_IMPL = config['aws']['spark.hadoop.fs.s3.impl']

# 初始化连接
spark = SparkSession \
    .builder \
    .appName('xtb-test') \
    .config("spark.hadoop.fs.s3a.access.key", AWS_ACCESS_KEY_ID) \
    .config("spark.hadoop.fs.s3a.secret.key", AWS_SECRET_ACCESS_KEY) \
    .config("spark.hadoop.fs.s3a.endpoint", AWS_ENDPOINT_ADDRESS) \
    .config("spark.hadoop.fs.s3.impl", AWS_IMPL) \
    .getOrCreate()

# 读表文件
df = spark \
    .read \
    .option("header", "true") \
    .option("recursiveFileLookup", "true") \
    .json("s3://stonewise-zinc-etl/new_database/hpc_test/rongfei_xtb_xyz_3/")
df.show(1)
df.printSchema()

# dataframe转rdd操作
# 做partition有两个原因：
# 1.让数据处理的数据，尽量均匀分给所有worker，让worker利用率最大，减少stage间隔中的等待时间
# 2.数据在仓库里存储最好分多个区，方便并行计算多个worker取用，降低数据倾斜的风险，减少shuffle这种耗时操作的可能（默认情况下1个worker拿一到多个分区数据）
data = df \
    .repartition(25) \
    .rdd \
    .map(lambda x: (x["file_name"], xtb_default(x["xyz"]))) \
 \
    # 以表的形式存储到hdfs（或者数仓s3地址）
schema = StructType([StructField("file_name", StringType(), True), StructField("result", StringType(), True)])
output = spark.createDataFrame(data, schema)
output.show(1)
output.printSchema()

output \
    .write \
    .mode("overwrite") \
    .parquet("s3://stonewise-zinc-etl/ww_test/test_xtb/monkey2/")

spark.stop()