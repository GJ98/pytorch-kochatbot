import sys
sys.path.append("../../")

import pymysql
from config.DatabaseConfig import DB

db = None
try:
    db = pymysql.connect(host=DB['db_host'],
                         user=DB['db_user'],
                         passwd=DB['db_password'],
                         db=DB['db_name'],
                         charset='utf8')

    sql = '''
      CREATE TABLE IF NOT EXISTS `chatbot_train_data` (
      `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
      `intent` VARCHAR(45) NULL,
      `ner` VARCHAR(1024) NULL,
      `query` TEXT NULL,
      `answer` TEXT NOT NULL,
      `answer_image` VARCHAR(2048) NULL,
      PRIMARY KEY (`id`))
    ENGINE = InnoDB DEFAULT CHARSET=utf8
    ''' 

    with db.cursor() as cursor:
        cursor.execute(sql)
    
except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()

