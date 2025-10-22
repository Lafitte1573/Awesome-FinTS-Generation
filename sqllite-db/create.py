import json
import sqlite3
import pandas as pd
from datetime import datetime
import os


class LocalDatabase:
    def __init__(self, db_name="local_database.db"):
        self.db_name = db_name
        self.connection = None
        self.cursor = None

    def connect(self):
        """连接到SQLite数据库"""
        try:
            self.connection = sqlite3.connect(self.db_name)
            self.cursor = self.connection.cursor()
            print(f"成功连接到数据库: {self.db_name}")
        except sqlite3.Error as e:
            print(f"数据库连接错误: {e}")

    def create_table(self):
        """创建示例数据表"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            year INTEGER,
            author TEXT NOT NULL,
            conference TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            theme TEXT NOT NULL,
            task TEXT NOT NULL,
            note TEXT NOT NULL
        );
        """
        try:
            self.cursor.executescript(create_table_sql)
            self.connection.commit()
            print("数据表创建成功")
        except sqlite3.Error as e:
            print(f"创建表错误: {e}")

    def insert_sample_data(self):
        """插入示例数据"""
        users_data = [
            ('张三', 'zhangsan@example.com', 25),
            ('李四', 'lisi@example.com', 30),
            ('王五', 'wangwu@example.com', 28),
            ('赵六', 'zhaoliu@example.com', 35)
        ]

        products_data = [
            ('笔记本电脑', 5999.99, '电子产品', 50),
            ('智能手机', 2999.99, '电子产品', 100),
            ('书籍', 49.99, '文化用品', 200),
            ('咖啡', 29.99, '食品', 150)
        ]

        try:
            # 插入用户数据
            self.cursor.executemany(
                "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
                users_data
            )

            # 插入产品数据
            self.cursor.executemany(
                "INSERT INTO products (name, price, category, stock) VALUES (?, ?, ?, ?)",
                products_data
            )

            self.connection.commit()
            print("示例数据插入成功")
        except sqlite3.Error as e:
            print(f"插入数据错误: {e}")

    def query_data(self, table_name):
        """查询指定表的所有数据"""
        try:
            self.cursor.execute(f"SELECT * FROM {table_name}")
            rows = self.cursor.fetchall()

            # 获取列名
            columns = [description[0] for description in self.cursor.description]

            # 使用pandas显示数据
            df = pd.DataFrame(rows, columns=columns)
            print(f"\n{table_name}表数据:")
            print(df)
            return df
        except sqlite3.Error as e:
            print(f"查询错误: {e}")
            return None

    def add_paper(self, name, email, age):
        """添加新用户"""
        try:
            self.cursor.execute(
                "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
                (name, email, age)
            )
            self.connection.commit()
            print(f"用户 {name} 添加成功")
        except sqlite3.Error as e:
            print(f"添加用户错误: {e}")

    def update_note(self, product_id, new_stock):
        """更新产品库存"""
        try:
            self.cursor.execute(
                "UPDATE products SET stock = ? WHERE id = ?",
                (new_stock, product_id)
            )
            self.connection.commit()
            print(f"产品ID {product_id} 库存更新为 {new_stock}")
        except sqlite3.Error as e:
            print(f"更新库存错误: {e}")

    def export_to_csv(self, table_name, filename=None):
        """将表数据导出为CSV文件"""
        if filename is None:
            filename = f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        df = self.query_data(table_name)
        if df is not None:
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"数据已导出到 {filename}")

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            print("数据库连接已关闭")


if __name__ == "__main__":
    # 创建数据库实例
    db = LocalDatabase("my_database.db")

    # 连接数据库
    connection = sqlite3.connect("my_database.db")
    cursor = connection.cursor()

    # 查询现有的表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    existing_tables = [table[0] for table in cursor.fetchall() if table[0] != 'sqlite_sequence']

    # 删除papers表（如果存在）
    for tb in existing_tables:
        print(tb)
        cursor.execute(f"DROP TABLE {tb};")
        print(f"已删除 {tb} 表")

    # 创建数据表 papers 和 notes
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS paper_metas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            authors TEXT NOT NULL,
            year INTEGER,
            theme TEXT NOT NULL,
            task TEXT NOT NULL,
            technique TEXT NOT NULL,
            abstract TEXT NOT NULL,
            method TEXT NOT NULL,
            paper_id TEXT NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS paper_cites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            name TEXT,
            paper_id TEXT NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS paper_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            note TEXT NOT NULL,
            paper_id TEXT NOT NULL
        );
    """)

    papers_data = []
    for i, file in enumerate(os.listdir('../assets/notes')):
        line = json.load(open(os.path.join('../assets/notes', file), 'r'))
        papers_data.append((
            i+1,
            line['title'],
            ', '.join(line['authors']),
            line['year'],
            line['research_directions'],
            line['specific_task'],
            ', '.join(line['techniques']),
            line['summary'],
            line['method'],
            file.split('.')[0]
        ))

    print(len(papers_data), papers_data[0])

    nick_name = pd.read_csv('../assets/nick_names.csv')
    paper_cites = []
    for i, line in nick_name.iterrows():
        assert line['paper_id'], "No paper_id"
        paper_cites.append((i+1, line['title'], line['nick_name'] if line['nick_name'] else 'None', line['paper_id']))
    print(len(paper_cites), paper_cites[0])

    notes_data = []
    for file in os.listdir('../assets/tech-notes'):
        if file.endswith('.md'):
            with open(os.path.join('../assets/tech-notes', file), 'r') as f:
                content = f.read()
                title = content.split('\n')[0].split('# ')[-1].strip()
                notes_data.append((len(notes_data)+1, title, content, file.rstrip('.md')))
    print(len(notes_data), notes_data[0:5])

    cursor.executemany(
        "INSERT INTO paper_cites (id, title, name, paper_id) VALUES (?, ?, ?, ?)",
        paper_cites
    )

    cursor.executemany(
        "INSERT INTO paper_metas (id, title, authors, year, theme, task, technique, abstract, method, paper_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        papers_data
    )

    cursor.executemany(
        "INSERT INTO paper_notes (id, title, note, paper_id) VALUES (?, ?, ?, ?)",
        notes_data
    )

    connection.commit()  # 提交更改
    print("数据已插入到数据库中")

    connection.close()  # 关闭数据库连接
