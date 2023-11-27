import pymysql

HOST = "gosu16954.rwlb.rds.aliyuncs.com"
DATABASE = "gosu_db"

# ZYH
USER1 = "gosu"
PWD1 = "zyhZYH123123!"
# CC
USER2 = "cyq"
PWD2 = "cyq102977!"

# 创建连接
conn2 = pymysql.connect(host=HOST, user=USER1, password=PWD1, database=DATABASE)
conn1 = pymysql.connect(host=HOST, user=USER2, password=PWD2, database=DATABASE)

# 创建游标
cursor = conn1.cursor()

# 执行SQL，并返回收影响行数
sql = "select * from user limit 10"
cursor.execute(sql)
# 打印结果
res = cursor.fetchall()
print(res)

# 提交，不然无法保存新建或者修改的数据
conn1.commit()

# 关闭游标
cursor.close()

# 关闭连接
conn1.close()