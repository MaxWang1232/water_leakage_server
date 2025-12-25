source activate water_leakage
export CUDA_VISIBLE_DEVICES=0
gunicorn --config=gun.py mysql_main:app