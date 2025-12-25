source activate water_leakage
export CUDA_VISIBLE_DEVICES=0
gunicorn --config=water_leakage_gun.py water_leakage_main:app