# sql_agent.py
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain_community.utilities import SQLDatabase

from typing import Literal

from langchain.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_community.agent_toolkits import SQLDatabaseToolkit

# 加载 .env 文件
load_dotenv()


# ===== 模型配置 =====
chat_model = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")  # 可选，默认使用 OpenAI 官方地址
)

support_model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen3:0.6b"),
    temperature=0
)

# ===== 数据库连接 =====
db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://{os.getenv('POSTGRESQL_USERNAME')}:{os.getenv('POSTGRESQL_PASSWORD')}@{os.getenv('POSTGRESQL_HOST')}:{os.getenv('POSTGRESQL_PORT')}/{os.getenv('POSTGRESQL_DBNAME')}",
    include_tables=[os.getenv("DB_TABLE_NAME", "parameter_tables_parameter")]
)

toolkit = SQLDatabaseToolkit(db=db, llm=chat_model)
tools = toolkit.get_tools()

# ===== 节点函数 =====
def list_tables(state: MessagesState):
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")

    return {"messages": [tool_call_message, tool_message, response]}

get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

def call_get_schema(state: MessagesState):
    llm_with_tools = chat_model.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ===== LLM 提示词 =====
generate_query_system_prompt = (
    f"You are an agent designed to interact with a SQL database.\n"
    f"Given an input question, create a syntactically correct {db.dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 50 results.\n"
    f"You can order the results by a relevant column to return the most interesting examples in the database. Never query for all the columns from a specific table, only ask for the relevant columns given the question.\n"
    f"You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.\n"
    f"DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.\n"
    f"Then you should query the schema of the most relevant tables.\n"
)

# ===== 表字段 schema 描述 =====
table_schema = {
    'id': '唯一ID',
    'carline': '车型项目代号',
    'project': '硬件版本',
    'parameter_name': '原始参数名称',
    'parameter_group': '参数所属分组',
    'data_type': '参数数据类型',
    'size': '数据大小（字节数）',
    'min_value': '参数最小值',
    'max_value': '参数最大值',
    'offset_value': '参数偏移值',
    'resolution': '参数分辨率',
    'dec_value': '参数小数位精度',
    'default_value': '参数默认值',
    'unit': '参数单位',
    'description': '参数描述或备注',
    'excel_file_id': '来源文件的 ID'
}

search_instruction = (
    "The database contains raw parameter information for vehicle models.\n"
    "Here is the schema you should consider:\n\n"
    f"{table_schema}\n"
    "Given a user's question, analyze the schema to determine which columns are relevant, "
    "and return only the necessary data to answer the question. "
    "If possible, return results formatted as a Markdown table."
)

run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")

def generate_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt,
    }
    llm_with_tools = chat_model.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message] + [search_instruction] + state["messages"])
    return {"messages": [response]}

check_query_system_prompt = """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.

You will call the appropriate tool to execute the query after running this check.
""".format(dialect=db.dialect)

def check_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": check_query_system_prompt,
    }
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    llm_with_tools = chat_model.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id
    return {"messages": [response]}

def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "check_query"

# ===== LangGraph 构建 =====
builder = StateGraph(MessagesState)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node('get_schema', ToolNode([get_schema_tool]))
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node('run_query', ToolNode([run_query_tool]))

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges("generate_query", should_continue)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")

graph = builder.compile()

