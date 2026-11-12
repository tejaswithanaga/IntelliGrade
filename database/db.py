
import mysql.connector
from config import Config

def get_connection():
    try:
        connection = mysql.connector.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DATABASE,
            port=Config.MYSQL_PORT
        )

        return connection

    except mysql.connector.Error as error:
        print("Database Connection Error:", error)
        return None


if __name__ == "__main__":
    connection = get_connection()

    if connection:
        print("MySQL Connected Successfully!")
        print("Database: intelligrade")
        connection.close()
    else:
        print("MySQL Connection Failed!")
