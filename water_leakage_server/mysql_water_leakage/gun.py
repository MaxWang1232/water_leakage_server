
import logging
import logging.handlers
import os
import multiprocessing
import gevent.monkey


gevent.monkey.patch_all()
bind = '0.0.0.0:11027'

# 2. 工作目录
chdir = '/home/niii/water_leakage_server/mysql_water_leakage'


# 4. 超时时间（保留60秒，适配模型预测+Parquet读写；若遇超时可改为120）
timeout = 60

# 5. 工作模式（不变，gevent异步模式提升并发）
worker_class = 'gthread'
workers = 4
threads = 3
thread_name_prefix = 'water-leak-'  # 线程名前缀，便于日志调试

loglevel = "info"

# 访问日志格式（不变，记录关键请求信息）
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s "%(f)s" "%(a)s"'

# PID文件（不变，用于管理服务进程）
pidfile = os.path.join(chdir, "water_gunicorn.pid")  # 确保PID文件生成在工作目录下

# 日志文件路径（优化：用绝对路径，避免路径混乱）
accesslog = os.path.join(chdir, "water_access.log")  # 访问日志（如请求IP、路径）
errorlog = os.path.join(chdir, "water_error.log")    # 错误日志（如代码异常、崩溃）
daemon = True
