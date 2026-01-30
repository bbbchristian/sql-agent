from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.llms.fake import FakeListLLM
from langchain.messages import AIMessage
import sqlite3

# 1. 创建示例数据库
def create_sample_db():
    conn = sqlite3.connect("sample.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        product TEXT
    )
    """)
    cursor.executemany("INSERT INTO users (name, age) VALUES (?, ?)", [
        ("Alice", 25),
        ("Bob", 30),
        ("Charlie", 35)
    ])
    cursor.executemany("INSERT INTO orders (user_id, product) VALUES (?, ?)", [
        (1, "Laptop"),
        (2, "Phone"),
        (3, "Tablet")
    ])
    conn.commit()
    conn.close()
    print("✅ sample.db 创建完成")

create_sample_db()

# 2. 连接数据库
db = SQLDatabase.from_uri("sqlite:///sample.db")

# 3. 创建一个假的 LLM（不会调用 API）
fake_llm = FakeListLLM(responses=["This is a fake LLM response"])

# 4. 创建 Toolkit（传入 fake_llm）
toolkit = SQLDatabaseToolkit(db=db, llm=fake_llm)
tools = toolkit.get_tools()

print("=== 可用工具 ===")
for tool in tools:
    print(f"{tool.name} - {tool.description}")

# 5. 调用 list_tables 工具
list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")

tool_call = {
    "name": "sql_db_list_tables",
    "args": {},
    "id": "abc123",
    "type": "tool_call",
}

tool_call_message = AIMessage(content="", tool_calls=[tool_call])
tool_message = list_tables_tool.invoke(tool_call)
response = AIMessage(f"Available tables: {tool_message.content}")

print("\n=== 工具调用过程 ===")
print("工具调用消息:", tool_call_message)
print("工具返回结果:", tool_message.content)
print("AI 回复:", response.content)
