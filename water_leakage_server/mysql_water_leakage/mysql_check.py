from mysql_water_leakage.mysql_config import get_mysql_connection

def check_database():
    mysql = get_mysql_connection()
    conn = mysql.get_connection()
    cursor = conn.cursor()
    
    try:
        # 检查数据量
        cursor.execute("SELECT COUNT(*) FROM water_data")
        count = cursor.fetchone()[0]
        print(f"当前数据库记录数: {count}")
        
        # 检查前几条数据
        cursor.execute("SELECT * FROM water_data LIMIT 5")
        rows = cursor.fetchall()
        if rows:
            print("前5条数据:")
            for row in rows:
                print(row)
        else:
            print("数据库为空")
            
    except Exception as e:
        print(f"检查数据库时出错: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    check_database()