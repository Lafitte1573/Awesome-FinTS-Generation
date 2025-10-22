import sqlite3
import pandas as pd
from datetime import datetime
import os


def db_query(sql_query):
    connection = sqlite3.connect("my_database.db")
    cursor = connection.cursor()
    cursor.execute(sql_query)
    rows = cursor.fetchall()
    connection.close()
    return rows


def db_update(sql_query):
    connection = sqlite3.connect("my_database.db")
    cursor = connection.cursor()
    cursor.execute(sql_query)
    connection.commit()
    connection.close()


if __name__ == "__main__":
    md_dir = '../markdowns'
    file_to_path = {}
    for file in os.listdir(md_dir):
        if file.endswith('.md'):
            with open(os.path.join(md_dir, file), 'r') as f:
                title = f.readlines()[0].split('# ')[-1].strip()
                # assert title not in file_to_path, f"{title} 已存在！"
                file_to_path[title] = os.path.join(md_dir, file)
    print(f"共找到 {len(file_to_path)} 篇文章的 Markdown 文件。")
    # for k, v in file_to_path.items():
    #     print(k, v)
    # exit()

    # # 连接数据库
    # connection = sqlite3.connect("my_database.db")
    # cursor = connection.cursor()

    # 检查没有题目的记录
    res = db_query(f"""
            SELECT paper_id
            FROM paper_metas
            WHERE title = 'Article';
        """)
    anonymous = [
        (r[0], "Concatenation Augmentation for Improving Deep Learning Models in Finance NLP with Scarce Data") if r[0] == 'electronics-14-02289'
        else (r[0], r[0].replace('_', ' '))
        for r in res
    ]

    if len(anonymous) > 0:
        print(anonymous)
        for paper_id, new_name in anonymous:
            db_update(f"""
                UPDATE paper_metas
                SET title = '{new_name}'
                WHERE paper_id = '{paper_id}';
            """)
            db_update(f"""
                UPDATE paper_cites
                SET title = '{new_name}'
                WHERE paper_id = '{paper_id}';
            """)
            # cursor.execute(f"""
            #     UPDATE paper_notes
            #     SET title = '{new_name}'
            #     WHERE paper_id = '{paper_id}';
            # """)

    # 查询现有的表
    themes = ['金融时序预测', '金融时序数据插补', '金融时序数据增强', '其他']
    en_themes = ['ftsf', 'fimp', 'ftsa', 'others']
    papers_data = {k: [] for k in themes}
    md_text = ''
    for theme, en_theme in zip(themes, en_themes):
        rows = []
        already_tts = []

        if not os.path.exists(f'annotations/{en_theme}_method_annotations.csv'):
            continue
        with open(f'annotations/{en_theme}_method_annotations.csv', 'r') as f:
            a_rows = pd.read_csv(f)
            titles = set(a_rows['title'].tolist())
            print(f"{theme} with {len(titles)} existing results")

        for tt in titles:
            line = db_query(f"""
                SELECT DISTINCT pm.title, pm.authors, pm.year, pm.abstract, pm.task, pc.name
                FROM paper_metas AS pm
                JOIN paper_cites AS pc ON pm.paper_id = pc.paper_id 
                WHERE pm.title='{tt}';
            """)
            if len(line) > 0 and tt not in already_tts:
                already_tts.append(tt)
                rows.append(line[0])

        # cursor.execute(f"""
        #     SELECT DISTINCT(p.title), p.author, p.year, pn.note
        #     FROM papers AS p
        #     JOIN paper_metas AS pm ON p.title = pm.title
        #     JOIN paper_notes AS pn ON p.title = pn.title
        #     WHERE pm.theme = '{theme}' AND NOT pm.is_survey;
        # """)
        # rows = cursor.fetchall()
        for r in db_query(f"""
            SELECT DISTINCT pm.title, pm.authors, pm.year, pm.abstract, pm.task, pc.name
            FROM paper_metas AS pm
            JOIN paper_cites AS pc ON pm.paper_id = pc.paper_id
            WHERE pm.theme='{theme}' AND pm.year>2022;
        """):
            if r[0] in already_tts:
                continue
            already_tts.append(r[0])
            rows += [r]
        print(len(rows), rows[:5])
        print(f"\n\n========= {theme} with {len(rows)} results =========\n")
        md_text += f"""
### {theme}
"""
        # 打印查询结果
        for i, row in enumerate(rows):
            if row[-1] is None:
                line = f"""#### {row[0]}
- **Authors**: {row[1]}
- **Year**: {row[2]}
- **Task**: {row[-2]}
- **Abstract**: {row[3]}
"""
                papers_data[theme].append(f"{row[1].split(',')[0]} et al. ({row[2]})")
            else:
                line = f"""#### [[{row[-1]}]]() {row[0]}
- **Authors**: {row[1]}
- **Year**: {row[2]}
- **Task**: {row[-2]}
- **Abstract**: {row[3]}
"""
                papers_data[theme].append(row[-1])

            md_text += '\n' + line
            print(i, f"{row[-1]}:", f'{row[0]}. {row[1].split(",")[0].split(" ")[-1]} et al. ({row[2]})')

        with open(f'tmp-output.md', 'w', encoding='utf-8') as f:
            f.write(md_text)

        notes = []
        for row in rows:
            # print(r[0])
            note = db_query(f"""
                        SELECT note
                        FROM paper_notes
                        WHERE title='{row[0]}'
                    """)
            # print(note)
            # exit()
            if len(note) > 0:
                notes.append(note[0][0])

        with open(f'{en_theme}_notes.md', 'w', encoding='utf-8') as f:
            f.write("\n\n ------------ \n\n".join(notes))

    print("\n\n========= All =========\n")
    for k, v in papers_data.items():
        print(k, len(v), v)

