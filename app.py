import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import json
from datetime import datetime
from zhipuai import ZhipuAI

# 尝试导入rpy2，如果失败则跳过RDS文件支持
try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    RPY2_AVAILABLE = True
except Exception as e:
    RPY2_AVAILABLE = False
    st.warning("R语言未安装，RDS文件支持已禁用。请安装R语言以启用RDS文件支持。")

# 配置页面设置
st.set_page_config(
    page_title="生物统计智能代码生成工具",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置黑金色主题
st.markdown("""
<style>
    .reportview-container {
        background: #0a0a0a;
        color: #e6c200;
    }
    .sidebar .sidebar-content {
        background: #1a1a1a;
        color: #e6c200;
    }
    .Widget>label {
        color: #e6c200;
    }
    .st-bk {
        background-color: #1a1a1a;
    }
    .st-at {
        color: #e6c200;
    }
    .st-ae {
        color: #e6c200;
    }
    .st-ag {
        background-color: #1a1a1a;
    }
    .st-ai {
        color: #e6c200;
    }
    .st-b3 {
        color: #e6c200;
    }
    .st-b4 {
        background-color: #1a1a1a;
    }
    .st-b5 {
        color: #e6c200;
    }
    .st-b6 {
        background-color: #1a1a1a;
    }
    .st-b7 {
        color: #e6c200;
    }
    .st-b8 {
        background-color: #1a1a1a;
    }
    .st-b9 {
        color: #e6c200;
    }
    .st-ba {
        background-color: #1a1a1a;
    }
    .st-bb {
        color: #e6c200;
    }
    .st-bc {
        background-color: #1a1a1a;
    }
    .st-bd {
        color: #e6c200;
    }
    .st-be {
        background-color: #1a1a1a;
    }
    .st-bf {
        color: #e6c200;
    }
    .st-bg {
        background-color: #1a1a1a;
    }
    .st-bh {
        color: #e6c200;
    }
    .st-bi {
        background-color: #1a1a1a;
    }
    .st-bj {
        color: #e6c200;
    }
    .st-bk {
        background-color: #1a1a1a;
    }
    .st-bl {
        color: #e6c200;
    }
    .st-bm {
        background-color: #1a1a1a;
    }
    .st-bn {
        color: #e6c200;
    }
    .st-bo {
        background-color: #1a1a1a;
    }
    .st-bp {
        color: #e6c200;
    }
    .st-bq {
        background-color: #1a1a1a;
    }
    .st-br {
        color: #e6c200;
    }
    .st-bs {
        background-color: #1a1a1a;
    }
    .st-bt {
        color: #e6c200;
    }
    .st-bu {
        background-color: #1a1a1a;
    }
    .st-bv {
        color: #e6c200;
    }
    .st-bw {
        background-color: #1a1a1a;
    }
    .st-bx {
        color: #e6c200;
    }
    .st-by {
        background-color: #1a1a1a;
    }
    .st-bz {
        color: #e6c200;
    }
    .st-c0 {
        background-color: #1a1a1a;
    }
    .st-c1 {
        color: #e6c200;
    }
    .st-c2 {
        background-color: #1a1a1a;
    }
    .st-c3 {
        color: #e6c200;
    }
    .st-c4 {
        background-color: #1a1a1a;
    }
    .st-c5 {
        color: #e6c200;
    }
    .st-c6 {
        background-color: #1a1a1a;
    }
    .st-c7 {
        color: #e6c200;
    }
    .st-c8 {
        background-color: #1a1a1a;
    }
    .st-c9 {
        color: #e6c200;
    }
    .st-ca {
        background-color: #1a1a1a;
    }
    .st-cb {
        color: #e6c200;
    }
    .st-cc {
        background-color: #1a1a1a;
    }
    .st-cd {
        color: #e6c200;
    }
    .st-ce {
        background-color: #1a1a1a;
    }
    .st-cf {
        color: #e6c200;
    }
    .st-d0 {
        background-color: #1a1a1a;
    }
    .st-d1 {
        color: #e6c200;
    }
    .st-d2 {
        background-color: #1a1a1a;
    }
    .st-d3 {
        color: #e6c200;
    }
    .st-d4 {
        background-color: #1a1a1a;
    }
    .st-d5 {
        color: #e6c200;
    }
    .st-d6 {
        background-color: #1a1a1a;
    }
    .st-d7 {
        color: #e6c200;
    }
    .st-d8 {
        background-color: #1a1a1a;
    }
    .st-d9 {
        color: #e6c200;
    }
    .st-da {
        background-color: #1a1a1a;
    }
    .st-db {
        color: #e6c200;
    }
    .st-dc {
        background-color: #1a1a1a;
    }
    .st-dd {
        color: #e6c200;
    }
    .st-de {
        background-color: #1a1a1a;
    }
    .st-df {
        color: #e6c200;
    }
    .st-e0 {
        background-color: #1a1a1a;
    }
    .st-e1 {
        color: #e6c200;
    }
    .st-e2 {
        background-color: #1a1a1a;
    }
    .st-e3 {
        color: #e6c200;
    }
    .st-e4 {
        background-color: #1a1a1a;
    }
    .st-e5 {
        color: #e6c200;
    }
    .st-e6 {
        background-color: #1a1a1a;
    }
    .st-e7 {
        color: #e6c200;
    }
    .st-e8 {
        background-color: #1a1a1a;
    }
    .st-e9 {
        color: #e6c200;
    }
    .st-ea {
        background-color: #1a1a1a;
    }
    .st-eb {
        color: #e6c200;
    }
    .st-ec {
        background-color: #1a1a1a;
    }
    .st-ed {
        color: #e6c200;
    }
    .st-ee {
        background-color: #1a1a1a;
    }
    .st-ef {
        color: #e6c200;
    }
    .st-f0 {
        background-color: #1a1a1a;
    }
    .st-f1 {
        color: #e6c200;
    }
    .st-f2 {
        background-color: #1a1a1a;
    }
    .st-f3 {
        color: #e6c200;
    }
    .st-f4 {
        background-color: #1a1a1a;
    }
    .st-f5 {
        color: #e6c200;
    }
    .st-f6 {
        background-color: #1a1a1a;
    }
    .st-f7 {
        color: #e6c200;
    }
    .st-f8 {
        background-color: #1a1a1a;
    }
    .st-f9 {
        color: #e6c200;
    }
    .st-fa {
        background-color: #1a1a1a;
    }
    .st-fb {
        color: #e6c200;
    }
    .st-fc {
        background-color: #1a1a1a;
    }
    .st-fd {
        color: #e6c200;
    }
    .st-fe {
        background-color: #1a1a1a;
    }
    .st-ff {
        color: #e6c200;
    }
    code {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        color: #e6c200;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #e6c200;
    }
    .stButton>button {
        background-color: #e6c200;
        color: #0a0a0a;
        border-radius: 4px;
    }
    .stTextInput>div>div>input {
        background-color: #1a1a1a;
        color: #e6c200;
    }
    .stTextArea>div>div>textarea {
        background-color: #1a1a1a;
        color: #e6c200;
    }
    .stSelectbox>div>div>select {
        background-color: #1a1a1a;
        color: #e6c200;
    }
    .stRadio>div>label {
        color: #e6c200;
    }
    .stFileUploader>div>div>label {
        color: #e6c200;
    }
    .stSuccess {
        color: #e6c200;
        background-color: rgba(230, 194, 0, 0.1);
        border-left: 4px solid #e6c200;
    }
    .stError {
        color: #ff4d4d;
        background-color: rgba(255, 77, 77, 0.1);
        border-left: 4px solid #ff4d4d;
    }
    .stInfo {
        color: #e6c200;
        background-color: rgba(230, 194, 0, 0.1);
        border-left: 4px solid #e6c200;
    }
</style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None
if 'requirements' not in st.session_state:
    st.session_state.requirements = ""
if 'code_language' not in st.session_state:
    st.session_state.code_language = "Python"
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ""
if 'execution_result' not in st.session_state:
    st.session_state.execution_result = None
if 'execution_output' not in st.session_state:
    st.session_state.execution_output = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# 创建历史记录目录
if not os.path.exists('history'):
    os.makedirs('history')

# 加载历史记录
def load_history():
    history_file = 'history/history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except:
                return []
    return []

# 保存历史记录
def save_history():
    history_file = 'history/history.json'
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)

# 侧边栏：项目标题和API密钥设置
with st.sidebar:
    st.title("🧬 生物统计智能助手")
    st.markdown("---")
    
    # API密钥设置
    st.subheader("智谱GLM-5 API设置")
    api_key = st.text_input("输入API密钥", value=st.session_state.api_key, type="password")
    if api_key:
        st.session_state.api_key = api_key
        st.success("API密钥已设置")
    
    st.markdown("---")

    # 模块1：数据集上传与预览
    st.subheader("📁 数据集上传")
    # 根据rpy2可用性设置支持的文件类型
    file_types = ["csv", "tsv", "xlsx"]
    if RPY2_AVAILABLE:
        file_types.append("rds")
    uploaded_file = st.file_uploader("选择数据文件", type=file_types)
    
    if uploaded_file:
        # 读取数据集
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension in ['.csv']:
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['.tsv']:
                df = pd.read_csv(uploaded_file, sep='\t')
            elif file_extension in ['.xlsx']:
                df = pd.read_excel(uploaded_file)
            elif file_extension in ['.rds'] and RPY2_AVAILABLE:
                # 使用rpy2读取RDS文件
                with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
                    df = robjects.r['readRDS'](uploaded_file.name)
            
            st.session_state.dataset = df
            st.session_state.dataset_info = {
                "filename": uploaded_file.name,
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict()
            }
            
            st.success(f"数据集加载成功！\n行数: {df.shape[0]}\n列数: {df.shape[1]}")
            
            # 数据集预览
            st.subheader("数据集预览")
            st.dataframe(df.head())
            
        except Exception as e:
            st.error(f"加载数据集失败: {str(e)}")
    
    st.markdown("---")

    # 模块2：生物统计需求输入
    st.subheader("📝 统计需求")
    
    # 代码语言选择
    code_language = st.radio("代码语言", ["Python", "R"], index=0 if st.session_state.code_language == "Python" else 1)
    st.session_state.code_language = code_language
    
    # 快捷模板
    st.subheader("快捷模板")
    template = st.selectbox(
        "常用分析模板",
        [
            "自定义需求",
            "转录组差异分析",
            "生存分析",
            "相关性分析",
            "方差分析",
            "富集分析"
        ]
    )
    
    # 根据模板生成需求
    if template != "自定义需求":
        if template == "转录组差异分析":
            requirements = f"对这个基因表达矩阵做差异表达分析，用{code_language}，输出火山图"
        elif template == "生存分析":
            requirements = f"对这个数据集做生存分析，用{code_language}，输出生存曲线"
        elif template == "相关性分析":
            requirements = f"对这个数据集做相关性分析，用{code_language}，输出相关性热图"
        elif template == "方差分析":
            requirements = f"对这个数据集做方差分析，用{code_language}，输出箱线图"
        elif template == "富集分析":
            requirements = f"对这个基因列表做富集分析，用{code_language}，输出富集结果图"
    else:
        requirements = st.text_area("输入统计需求", value=st.session_state.requirements, height=150)
    
    st.session_state.requirements = requirements
    
    # 提交按钮
    submit_button = st.button("🚀 生成代码")

# 主区域
st.title("🧬 生物统计智能代码生成工具")
st.markdown("---")

# 模块3：GLM-5 API核心交互
if submit_button and st.session_state.dataset is not None and st.session_state.requirements:
    if not st.session_state.api_key:
        st.error("请先设置智谱GLM-5 API密钥")
    else:
        with st.spinner("正在调用GLM-5 API生成代码..."):
            try:
                # 初始化ZhipuAI客户端
                client = ZhipuAI(api_key=st.session_state.api_key)
                
                # 构建Prompt
                dataset_info = st.session_state.dataset_info
                prompt = f"""
你是一个专业的生物统计分析助手，擅长使用Python和R进行生物数据分析。

请根据以下数据集信息和用户需求，生成对应的{st.session_state.code_language}代码：

数据集信息：
- 文件名：{dataset_info['filename']}
- 数据形状：{dataset_info['shape']}
- 字段名：{dataset_info['columns']}
- 数据类型：{dataset_info['dtypes']}

用户需求：
{st.session_state.requirements}

请严格按照以下要求输出：
1. 仅输出完整的{st.session_state.code_language}代码，不要有任何解释性文字
2. 代码应包含数据读取、分析和结果可视化
3. 确保代码可以直接运行，并且输出结果和图表
4. 对于Python，请使用pandas、matplotlib、seaborn等常用库
5. 对于R，请使用ggplot2等常用库
6. 代码中请使用相对路径读取数据文件
7. 输出格式：仅代码，无其他内容
"""
                
                # 调用API
                response = client.chat.completions.create(
                    model="glm-5",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                # 获取生成的代码
                generated_code = response.choices[0].message.content.strip()
                st.session_state.generated_code = generated_code
                
                # 保存到历史记录
                history_item = {
                    "timestamp": datetime.now().isoformat(),
                    "filename": dataset_info['filename'],
                    "requirements": st.session_state.requirements,
                    "code_language": st.session_state.code_language,
                    "generated_code": generated_code
                }
                st.session_state.history.append(history_item)
                save_history()
                
                st.success("代码生成成功！")
                
            except Exception as e:
                st.error(f"API调用失败: {str(e)}")

# 模块4：代码展示与一键运行
if st.session_state.generated_code:
    st.subheader("💻 生成的代码")
    
    # 代码展示
    code_tab, result_tab = st.tabs(["代码", "运行结果"])
    
    with code_tab:
        st.code(st.session_state.generated_code, language=st.session_state.code_language.lower())
        
        # 一键复制代码
        if st.button("📋 复制代码"):
            st.write("代码已复制到剪贴板！")
        
        # 一键运行代码
        if st.button("▶️ 运行代码"):
            with st.spinner("正在运行代码..."):
                try:
                    # 保存代码到文件
                    code_file = f"generated_code.{st.session_state.code_language.lower()}"
                    with open(code_file, 'w', encoding='utf-8') as f:
                        f.write(st.session_state.generated_code)
                    
                    # 运行代码
                    if st.session_state.code_language == "Python":
                        # 执行Python代码
                        exec_globals = {}
                        exec_locals = {}
                        exec(st.session_state.generated_code, exec_globals, exec_locals)
                        st.session_state.execution_result = exec_locals
                        st.session_state.execution_output = "代码执行成功！"
                        st.success("代码运行成功！")
                    else:
                        # 执行R代码
                        if RPY2_AVAILABLE:
                            robjects.r(st.session_state.generated_code)
                            st.session_state.execution_output = "代码执行成功！"
                            st.success("代码运行成功！")
                        else:
                            st.error("R语言未安装，无法执行R代码。请安装R语言或选择Python代码。")
                    
                except Exception as e:
                    st.error(f"代码运行失败: {str(e)}")
    
    with result_tab:
        if st.session_state.execution_output:
            st.info(st.session_state.execution_output)
        
        # 展示运行结果
        if st.session_state.execution_result:
            st.subheader("运行结果")
            for key, value in st.session_state.execution_result.items():
                if isinstance(value, pd.DataFrame):
                    st.dataframe(value)
                elif isinstance(value, plt.Figure):
                    st.pyplot(value)
                elif isinstance(value, (np.ndarray, list, dict)):
                    st.write(value)

# 模块5：结果导出与历史记录
st.markdown("---")
st.subheader("📊 结果导出")

col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.generated_code:
        if st.button("💾 导出代码"):
            code_file = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{st.session_state.code_language.lower()}"
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(st.session_state.generated_code)
            st.success(f"代码已导出到: {code_file}")

with col2:
    if st.session_state.execution_result:
        if st.button("📈 导出结果"):
            result_file = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            # 导出第一个DataFrame结果
            for value in st.session_state.execution_result.values():
                if isinstance(value, pd.DataFrame):
                    value.to_csv(result_file, index=False, encoding='utf-8-sig')
                    st.success(f"结果已导出到: {result_file}")
                    break

with col3:
    if st.session_state.execution_result:
        if st.button("🖼️ 导出图表"):
            chart_file = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            # 导出第一个图表
            for value in st.session_state.execution_result.values():
                if isinstance(value, plt.Figure):
                    value.savefig(chart_file, dpi=300, bbox_inches='tight')
                    st.success(f"图表已导出到: {chart_file}")
                    break

# 历史记录
st.markdown("---")
st.subheader("📜 历史记录")

if st.session_state.history:
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"历史记录 {i+1} - {item['timestamp'][:19]}"):
            st.write(f"**文件名:** {item['filename']}")
            st.write(f"**需求:** {item['requirements']}")
            st.write(f"**语言:** {item['code_language']}")
            st.code(item['generated_code'], language=item['code_language'].lower())
else:
    st.info("暂无历史记录")

# 底部信息
st.markdown("---")
st.markdown("### 🧬 生物统计智能代码生成工具 v2.0")
st.markdown("- 基于智谱GLM-5大模型")
st.markdown("- 支持Python/R双语言")
st.markdown("- 极简深色科技风")
st.markdown("- 自用高效生物统计助手")
