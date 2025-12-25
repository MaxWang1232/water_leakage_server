pid=$(cat water_gunicorn.pid | awk '{print $1}' )
echo $pid
kill -9 $pid
rm -rf *.log water_gunicorn.pid