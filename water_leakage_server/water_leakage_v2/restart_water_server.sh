
#!/bin/bash

cd /home/niii/water_leakage_server/water_leakage_v2  || {
  echo "错误：服务目录不存在！"
  exit 1
}

# 停止服务（仅当pid文件存在时执行）
if [ -f "water_gunicorn.pid" ]; then
  pid=$(cat water_gunicorn.pid | awk '{print $1}')
  echo "停止服务进程：$pid"
  kill -9 $pid
  rm -rf *.log water_gunicorn.pid
  echo "旧进程已清理"
else
  echo "未找到water_gunicorn.pid，跳过停止步骤"
fi

sleep 2  # 等待进程退出

# 激活conda环境并启动服务
echo "激活环境并启动服务..."
source activate water_leakage || {
  echo "错误：激活water_leakage环境失败！"
  exit 1
}

export CUDA_VISIBLE_DEVICES=0

# 启动gunicorn（替换为你的gunicorn实际路径，可通过which gunicorn查看）
gunicorn_path=$(which gunicorn)
if [ -z "$gunicorn_path" ]; then
  echo "错误：未在当前环境找到gunicorn，请先安装！"
  exit 1
fi

$gunicorn_path --config=water_leakage_gun.py water_leakage_main:app || {
  echo "错误：服务启动失败！"
  exit 1
}

echo "服务启动成功"
