import logging
import os
import gevent.monkey

# 异步处理补丁（支持高并发连续请求）
gevent.monkey.patch_all()

# 服务绑定地址
bind = '0.0.0.0:11027'

# 工作目录（替换为实际项目路径）
chdir = '/home/niii/water_leakage_server/water_leakage_v2'

# 超时时间（应对连续请求，适当延长）
timeout = 120

# 工作模式与进程数（根据服务器CPU核心数调整，支持高并发）
worker_class = 'gthread'
workers = 4  # 建议设为CPU核心数的2-4倍
threads = 4  # 每个进程的线程数，增加并发处理能力

# 日志配置（记录连续请求详情）
loglevel = "info"
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s "%(f)s" "%(a)s"'
pidfile = os.path.join(chdir, "water_gunicorn.pid")
accesslog = os.path.join(chdir, "water_access.log")
errorlog = os.path.join(chdir, "water_error.log")

# 后台运行（生产环境启用，支持连续请求处理）
daemon = True