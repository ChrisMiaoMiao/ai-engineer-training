import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

# LlamaIndex 核心组件
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter, SentenceWindowNodeParser
from llama_index.core.schema import TextNode
from llama_index.core.postprocessor import MetadataReplacementPostProcessor

# LLM 和 Embedding 模型
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels



# 1. 加载环境变量和初始化全局配置
load_dotenv()


# 配置 LLM（大语言模型）- 使用阿里云通义千问
Settings.llm = OpenAILike(
    model="qwen-plus",  # 使用通义千问 Plus 模型
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True  # 指定为对话模型
)

# 配置 Embedding 模型（文本嵌入模型）- 用于将文本转换为向量
Settings.embed_model = DashScopeEmbedding(
    model=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,  # 使用 V3 版本的嵌入模型
    embed_batch_size=6,  # 批处理大小，每次处理 6 个文本
    embed_input_length=8192  # 最大输入长度为 8192 tokens
)

# 配置文本切片器（用于 SentenceWindowNodeParser）
# 这个设置会被 SentenceWindowNodeParser 内部使用来分割文本
Settings.text_splitter = SentenceSplitter(
    chunk_size=1024,  # 基础分句器的块大小
    chunk_overlap=20  # 基础分句器的重叠大小
)


# ============================================================
# 2. 从文件加载测试文档
# ============================================================
def load_documents_from_data_folder() -> List[Document]:
    """
    从 data 文件夹加载测试文档
    
    这个函数使用 SimpleDirectoryReader 从 data 文件夹中读取所有文本文件，
    并将它们转换为 LlamaIndex 的 Document 对象。
    
    Returns:
        包含从文件加载的文档列表
    
    Raises:
        ValueError: 如果 data 文件夹不存在或为空
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建 data 文件夹的绝对路径
    data_dir = os.path.join(current_dir, "data")
    
    # 检查 data 文件夹是否存在
    if not os.path.exists(data_dir):
        raise ValueError(f"❌ data 文件夹不存在: {data_dir}\n请先创建 data 文件夹并添加测试文档")
    
    # 检查 data 文件夹是否为空
    if not os.listdir(data_dir):
        raise ValueError(f"❌ data 文件夹为空: {data_dir}\n请添加至少一个测试文档")
    
    print(f"✓ 正在从 data 文件夹加载文档: {data_dir}")
    
    # 使用 SimpleDirectoryReader 读取文件夹中的所有文档
    # SimpleDirectoryReader 会自动识别多种文件格式（.txt, .pdf, .docx 等）
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        recursive=True,  # 递归读取子文件夹
        required_exts=[".txt"]  # 只读取 .txt 文件，可以根据需要修改
    )
    
    # 加载所有文档
    documents = reader.load_data()
    
    # 打印文档信息
    print(f"✓ 成功加载 {len(documents)} 个文档")
    for i, doc in enumerate(documents, 1):
        # 获取文件名
        filename = doc.metadata.get('file_name', 'Unknown')
        # 计算文档字符数
        char_count = len(doc.text)
        # 估算词数（中文按字符数，英文按空格分割）
        word_count = len(doc.text.split()) if doc.text.strip() else 0
        
        print(f"  文档 {i}: {filename}")
        print(f"    - 字符数: {char_count:,}")
        print(f"    - 估算词数: {word_count:,}")
        print(f"    - 前100字符预览: {doc.text[:100].strip()}...")
    
    return documents


# ============================================================
# 3. 评估函数 - 测试不同的切片参数（通用版本）
# ============================================================
def evaluate_splitter(
    splitter,  # 可以是 SentenceSplitter 或 TokenTextSplitter
    documents: List[Document],
    query: str,
    config_name: str
) -> Dict[str, Any]:
    """
    评估特定切片器配置的性能（支持句子切片和 Token 切片）
    
    Args:
        splitter: 切片器实例（SentenceSplitter 或 TokenTextSplitter）
        documents: 要索引的文档列表
        query: 测试查询问题
        config_name: 配置名称（用于标识）
    
    Returns:
        包含评估结果的字典
    """
    print(f"\n{'='*60}")
    print(f"测试配置: {config_name}")
    print(f"{'='*60}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 1. 使用句子切片器将文档分割成节点（chunks）
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"✓ 文档切片完成: 生成了 {len(nodes)} 个文本块（chunks）")
    
    # 显示前 3 个节点的信息
    print(f"\n前 3 个文本块示例:")
    for i, node in enumerate(nodes[:3]):
        print(f"\n  块 #{i+1}:")
        print(f"  - 长度: {len(node.text)} 字符")
        print(f"  - 预览: {node.text[:100]}...")
    
    # 2. 从节点创建向量索引
    print(f"\n✓ 开始创建向量索引...")
    index = VectorStoreIndex(nodes)
    print(f"✓ 向量索引创建完成")
    
    # 3. 创建查询引擎
    query_engine = index.as_query_engine(
        similarity_top_k=3  # 检索最相似的 3 个文本块
    )
    
    # 4. 执行查询
    print(f"\n✓ 执行查询: '{query}'")
    response = query_engine.query(query)
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 5. 显示结果
    print(f"\n【查询结果】")
    print(f"回答: {response.response}")
    print(f"\n【检索到的源文本块】")
    for i, source_node in enumerate(response.source_nodes):
        print(f"\n  相关文本块 #{i+1} (相似度分数: {source_node.score:.4f}):")
        print(f"  {source_node.text[:200]}...")
    
    # 6. 返回评估指标
    results = {
        "config_name": config_name,
        "num_chunks": len(nodes),
        "query_time": elapsed_time,
        "response": response.response,
        "num_sources": len(response.source_nodes),
        "avg_similarity": sum(node.score for node in response.source_nodes) / len(response.source_nodes) if response.source_nodes else 0
    }
    
    print(f"\n⏱️  总耗时: {elapsed_time:.2f} 秒")
    print(f"📊 平均相似度分数: {results['avg_similarity']:.4f}")
    
    return results


# ============================================================
# 3.2 评估函数 - 专门用于句子窗口切片
# ============================================================
def evaluate_sentence_window_splitter(
    splitter: SentenceWindowNodeParser,
    documents: List[Document],
    query: str,
    config_name: str
) -> Dict[str, Any]:
    """
    评估句子窗口切片器的性能
    
    句子窗口切片的特点：
    - 将文档按句子切分，每个句子作为一个节点
    - 在元数据中保存周围句子的上下文窗口
    - 检索时只用单句做匹配，但返回时可以包含周围上下文
    - 这种方法结合了精确检索和丰富上下文的优势
    
    Args:
        splitter: 句子窗口切片器实例
        documents: 要索引的文档列表
        query: 测试查询问题
        config_name: 配置名称（用于标识）
    
    Returns:
        包含评估结果的字典
    """
    print(f"\n{'='*60}")
    print(f"测试配置: {config_name}")
    print(f"{'='*60}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 关键步骤：先用 SentenceSplitter 手动分割文档为基础节点
    # 这样可以避免 SentenceWindowNodeParser 将整个文档当作一个句子
    print(f"✓ 步骤1: 使用 SentenceSplitter 预处理文档...")
    base_splitter = SentenceSplitter(
        chunk_size=512,  # 基础分割的块大小
        chunk_overlap=50  # 基础分割的重叠大小
    )
    base_nodes = base_splitter.get_nodes_from_documents(documents)
    print(f"✓ 预处理完成: 生成了 {len(base_nodes)} 个基础文本块")
    
    # 在基础节点上应用窗口策略
    print(f"✓ 步骤2: 在基础节点上构建句子窗口...")
    nodes = splitter.build_window_nodes_from_documents(base_nodes)
    print(f"✓ 窗口构建完成: 生成了 {len(nodes)} 个窗口节点")
    
    # 显示前 3 个节点的信息
    print(f"\n前 3 个句子窗口节点示例:")
    for i, node in enumerate(nodes[:3]):
        print(f"\n  窗口节点 #{i+1}:")
        print(f"  - 核心文本长度: {len(node.text)} 字符")
        print(f"  - 核心文本: {node.text[:100]}...")
        # 显示窗口上下文信息（如果有）
        if 'window' in node.metadata:
            window_text = node.metadata['window']
            print(f"  - 窗口上下文长度: {len(window_text)} 字符")
            print(f"  - 窗口上下文预览: {window_text[:150]}...")
    
    # 2. 从节点创建向量索引
    print(f"\n✓ 开始创建向量索引...")
    index = VectorStoreIndex(nodes)
    print(f"✓ 向量索引创建完成")
    
    # 3. 创建查询引擎，使用 MetadataReplacementPostProcessor
    # 这个后处理器会用窗口上下文替换检索到的节点文本
    query_engine = index.as_query_engine(
        similarity_top_k=3,  # 检索最相似的 3 个句子节点
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ]
    )
    
    # 4. 执行查询
    print(f"\n✓ 执行查询: '{query}'")
    response = query_engine.query(query)
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 5. 显示结果
    print(f"\n【查询结果】")
    print(f"回答: {response.response}")
    print(f"\n【检索到的源文本（包含窗口上下文）】")
    for i, source_node in enumerate(response.source_nodes):
        print(f"\n  相关句子 #{i+1} (相似度分数: {source_node.score:.4f}):")
        print(f"  {source_node.text[:300]}...")
    
    # 6. 返回评估指标
    results = {
        "config_name": config_name,
        "num_chunks": len(nodes),
        "query_time": elapsed_time,
        "response": response.response,
        "num_sources": len(response.source_nodes),
        "avg_similarity": sum(node.score for node in response.source_nodes) / len(response.source_nodes) if response.source_nodes else 0
    }
    
    print(f"\n⏱️  总耗时: {elapsed_time:.2f} 秒")
    print(f"📊 平均相似度分数: {results['avg_similarity']:.4f}")
    
    return results


# ============================================================
# 4. 主函数 - 测试不同的参数组合
# ============================================================
def main():
    """
    主测试函数：测试不同的句子切片参数对检索效果的影响
    """
    # 验证 API Key
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        print("❌ 错误: DASHSCOPE_API_KEY 未设置，请在 .env 文件中配置")
        return
    
    print(f"✓ DashScope API Key 已加载")
    print(f"\n{'#'*80}")
    print(f"# LlamaIndex 文本切片参数影响测试")
    print(f"# 包括：句子切片（SentenceSplitter）和 Token 切片（TokenTextSplitter）")
    print(f"{'#'*80}\n")
    
    # 从 data 文件夹加载测试文档
    try:
        documents = load_documents_from_data_folder()
    except ValueError as e:
        print(str(e))
        return
    
    print(f"\n✓ 文档加载完成，共 {len(documents)} 个文档")
    
    # 定义测试查询
    test_query = "什么是深度学习？它与机器学习有什么关系？"
    print(f"✓ 测试问题: {test_query}")
    
    # ============================================================
    # 第一部分：测试句子切片（SentenceSplitter）参数配置
    # ============================================================
    print(f"\n{'*'*80}")
    print(f"* 第一部分：句子切片（SentenceSplitter）测试")
    print(f"* 说明：按照句子边界进行切片，保持语义完整性")
    print(f"{'*'*80}")
    
    # 句子切片参数说明:
    # - chunk_size: 每个文本块的最大字符数
    # - chunk_overlap: 相邻文本块之间的重叠字符数
    # - paragraph_separator: 段落分隔符
    
    sentence_configurations = [
        {
            "name": "句子切片-配置1: 小块 + 无重叠",
            "chunk_size": 256,
            "chunk_overlap": 0
        },
        {
            "name": "句子切片-配置2: 小块 + 小重叠",
            "chunk_size": 256,
            "chunk_overlap": 50
        },
        {
            "name": "句子切片-配置3: 中等块 + 中等重叠",
            "chunk_size": 512,
            "chunk_overlap": 50
        },
        {
            "name": "句子切片-配置4: 中等块 + 大重叠",
            "chunk_size": 512,
            "chunk_overlap": 128
        },
        {
            "name": "句子切片-配置5: 大块 + 中等重叠",
            "chunk_size": 1024,
            "chunk_overlap": 100
        },
    ]
    
    # 存储所有测试结果
    all_results = []
    
    # 遍历每个句子切片配置进行测试
    for config in sentence_configurations:
        # 创建句子切片器
        splitter = SentenceSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            paragraph_separator="\n\n",  # 使用双换行作为段落分隔符
        )
        
        # 执行评估
        result = evaluate_splitter(
            splitter=splitter,
            documents=documents,
            query=test_query,
            config_name=config["name"]
        )
        
        all_results.append(result)
        
        # 在测试之间稍作延迟，避免 API 限流
        time.sleep(2)
    
    # ============================================================
    # 第二部分：测试 Token 切片（TokenTextSplitter）参数配置
    # ============================================================
    print(f"\n\n{'*'*80}")
    print(f"* 第二部分：Token 切片（TokenTextSplitter）测试")
    print(f"* 说明：按照 Token 数量进行切片，更精确地控制文本块大小")
    print(f"{'*'*80}")
    
    # Token 切片参数说明:
    # - chunk_size: 每个文本块的最大 Token 数量
    # - chunk_overlap: 相邻文本块之间的重叠 Token 数量
    # - separator: Token 分隔符（默认为空格）
    #
    # 注意：Token 切片与句子切片的主要区别：
    # 1. Token 切片按 token 数量切分，更精确地控制大小
    # 2. 句子切片尊重句子边界，保持语义完整性
    # 3. Token 切片适合对长度有严格要求的场景（如 API 限制）
    # 4. 句子切片适合保持上下文完整性的场景（如问答系统）
    
    token_configurations = [
        {
            "name": "Token切片-配置1: 小块(128 tokens) + 无重叠",
            "chunk_size": 128,
            "chunk_overlap": 0
        },
        {
            "name": "Token切片-配置2: 小块(128 tokens) + 小重叠(20 tokens)",
            "chunk_size": 128,
            "chunk_overlap": 20
        },
        {
            "name": "Token切片-配置3: 中等块(256 tokens) + 中等重叠(30 tokens)",
            "chunk_size": 256,
            "chunk_overlap": 30
        },
        {
            "name": "Token切片-配置4: 中等块(256 tokens) + 大重叠(50 tokens)",
            "chunk_size": 256,
            "chunk_overlap": 50
        },
        {
            "name": "Token切片-配置5: 大块(512 tokens) + 中等重叠(50 tokens)",
            "chunk_size": 512,
            "chunk_overlap": 50
        },
    ]
    
    # 遍历每个 Token 切片配置进行测试
    for config in token_configurations:
        # 创建 Token 切片器
        # TokenTextSplitter 使用分词器将文本分割成 tokens，然后按指定大小切片
        splitter = TokenTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separator=" ",  # 使用空格作为基本分隔符
        )
        
        # 执行评估
        result = evaluate_splitter(
            splitter=splitter,
            documents=documents,
            query=test_query,
            config_name=config["name"]
        )
        
        all_results.append(result)
        
        # 在测试之间稍作延迟，避免 API 限流
        time.sleep(2)
    
    # ============================================================
    # 第三部分：测试句子窗口切片（SentenceWindowNodeParser）参数配置
    # ============================================================
    print(f"\n\n{'*'*80}")
    print(f"* 第三部分：句子窗口切片（SentenceWindowNodeParser）测试")
    print(f"* 说明：将文档按句子切分，每个句子保存周围句子作为上下文窗口")
    print(f"*       检索时用单句匹配，返回时包含窗口上下文，兼顾精确性和完整性")
    print(f"{'*'*80}")
    
    # 句子窗口切片参数说明:
    # - window_size: 窗口大小，即在核心句子前后各保留多少个句子
    # - window_metadata_key: 存储窗口上下文的元数据键名
    # - original_text_metadata_key: 存储原始句子的元数据键名
    #
    # 工作原理：
    # 1. 使用 Settings.text_splitter (SentenceSplitter) 将文档按句子切分
    # 2. 为每个句子节点保存前后 N 个句子作为上下文窗口
    # 3. 向量化时只对核心句子进行嵌入（保证检索精确性）
    # 4. 返回结果时用窗口上下文替换核心句子（提供完整上下文）
    #
    # 优势：
    # - 检索精确：只匹配核心句子，不受无关上下文干扰
    # - 上下文丰富：返回时包含周围句子，便于理解
    # - 适合问答：可以精确定位答案句，同时提供足够的背景信息
    
    sentence_window_configurations = [
        {
            "name": "句子窗口-配置1: 窗口大小=1（前后各1句）",
            "window_size": 1
        },
        {
            "name": "句子窗口-配置2: 窗口大小=2（前后各2句）",
            "window_size": 2
        },
        {
            "name": "句子窗口-配置3: 窗口大小=3（前后各3句）",
            "window_size": 3
        },
        {
            "name": "句子窗口-配置4: 窗口大小=5（前后各5句）",
            "window_size": 5
        },
        {
            "name": "句子窗口-配置5: 窗口大小=10（前后各10句）",
            "window_size": 10
        },
    ]
    
    # 遍历每个句子窗口配置进行测试
    for config in sentence_window_configurations:
        # 创建句子窗口切片器
        # 注意：SentenceWindowNodeParser 会使用 Settings.text_splitter 来分割文本
        splitter = SentenceWindowNodeParser.from_defaults(
            window_size=config["window_size"],
            window_metadata_key="window",  # 窗口上下文存储在 'window' 元数据中
            original_text_metadata_key="original_sentence"  # 原始句子存储键
        )
        
        # 执行评估（使用专门的句子窗口评估函数）
        result = evaluate_sentence_window_splitter(
            splitter=splitter,
            documents=documents,
            query=test_query,
            config_name=config["name"]
        )
        
        all_results.append(result)
        
        # 在测试之间稍作延迟，避免 API 限流
        time.sleep(2)
    
    # ============================================================
    # 输出总结报告
    # ============================================================
    print(f"\n\n{'='*100}")
    print(f"测试总结报告 - 句子切片 vs Token 切片 vs 句子窗口切片 三方对比")
    print(f"{'='*100}\n")
    
    print(f"{'配置名称':<60} | {'文本块数':<10} | {'查询耗时(秒)':<14} | {'平均相似度':<12}")
    print(f"{'-'*60}-+-{'-'*10}-+-{'-'*14}-+-{'-'*12}")
    
    for result in all_results:
        print(f"{result['config_name']:<60} | {result['num_chunks']:<10} | "
              f"{result['query_time']:<14.2f} | {result['avg_similarity']:<12.4f}")
    
    # 分别找出三种切片方法的最佳配置
    sentence_results = [r for r in all_results if "句子切片-" in r['config_name']]
    token_results = [r for r in all_results if "Token切片" in r['config_name']]
    window_results = [r for r in all_results if "句子窗口" in r['config_name']]
    
    print(f"\n{'='*100}")
    print(f"最佳配置分析")
    print(f"{'='*100}")
    
    if sentence_results:
        best_sentence = max(sentence_results, key=lambda x: x['avg_similarity'])
        print(f"\n🏆 句子切片最佳配置: {best_sentence['config_name']}")
        print(f"   - 平均相似度分数: {best_sentence['avg_similarity']:.4f}")
        print(f"   - 生成文本块数: {best_sentence['num_chunks']}")
        print(f"   - 查询耗时: {best_sentence['query_time']:.2f} 秒")
    
    if token_results:
        best_token = max(token_results, key=lambda x: x['avg_similarity'])
        print(f"\n🏆 Token 切片最佳配置: {best_token['config_name']}")
        print(f"   - 平均相似度分数: {best_token['avg_similarity']:.4f}")
        print(f"   - 生成文本块数: {best_token['num_chunks']}")
        print(f"   - 查询耗时: {best_token['query_time']:.2f} 秒")
    
    if window_results:
        best_window = max(window_results, key=lambda x: x['avg_similarity'])
        print(f"\n🏆 句子窗口切片最佳配置: {best_window['config_name']}")
        print(f"   - 平均相似度分数: {best_window['avg_similarity']:.4f}")
        print(f"   - 生成文本块数: {best_window['num_chunks']}")
        print(f"   - 查询耗时: {best_window['query_time']:.2f} 秒")
    
    # 总体最佳配置
    overall_best = max(all_results, key=lambda x: x['avg_similarity'])
    print(f"\n🎯 总体最佳配置: {overall_best['config_name']}")
    print(f"   - 平均相似度分数: {overall_best['avg_similarity']:.4f}")
    print(f"   - 生成文本块数: {overall_best['num_chunks']}")
    print(f"   - 查询耗时: {overall_best['query_time']:.2f} 秒")
    
    # 关键洞察
    print(f"\n{'='*100}")
    print(f"关键洞察")
    print(f"{'='*100}")
    
    avg_sentence_chunks = sum(r['num_chunks'] for r in sentence_results) / len(sentence_results) if sentence_results else 0
    avg_token_chunks = sum(r['num_chunks'] for r in token_results) / len(token_results) if token_results else 0
    avg_window_chunks = sum(r['num_chunks'] for r in window_results) / len(window_results) if window_results else 0
    
    avg_sentence_similarity = sum(r['avg_similarity'] for r in sentence_results) / len(sentence_results) if sentence_results else 0
    avg_token_similarity = sum(r['avg_similarity'] for r in token_results) / len(token_results) if token_results else 0
    avg_window_similarity = sum(r['avg_similarity'] for r in window_results) / len(window_results) if window_results else 0
    
    print(f"\n📊 句子切片统计:")
    print(f"   - 平均生成文本块数: {avg_sentence_chunks:.1f}")
    print(f"   - 平均相似度分数: {avg_sentence_similarity:.4f}")
    
    print(f"\n📊 Token 切片统计:")
    print(f"   - 平均生成文本块数: {avg_token_chunks:.1f}")
    print(f"   - 平均相似度分数: {avg_token_similarity:.4f}")
    
    print(f"\n📊 句子窗口切片统计:")
    print(f"   - 平均生成文本块数: {avg_window_chunks:.1f}")
    print(f"   - 平均相似度分数: {avg_window_similarity:.4f}")
    
    print(f"\n💡 建议:")
    
    # 找出表现最好的方法
    method_scores = {
        "句子切片": avg_sentence_similarity,
        "Token 切片": avg_token_similarity,
        "句子窗口切片": avg_window_similarity
    }
    best_method = max(method_scores, key=method_scores.get)
    
    if best_method == "句子切片":
        print(f"   ✨ 句子切片在本次测试中表现最好（平均相似度: {avg_sentence_similarity:.4f}）")
        print(f"   📌 句子切片能更好地保持语义完整性，适合问答和对话系统。")
        print(f"   📌 适用场景：需要保持句子完整性和上下文连贯性的应用")
    elif best_method == "Token 切片":
        print(f"   ✨ Token 切片在本次测试中表现最好（平均相似度: {avg_token_similarity:.4f}）")
        print(f"   📌 Token 切片能更精确地控制文本块大小，适合有严格长度限制的场景。")
        print(f"   📌 适用场景：API token 限制、模型上下文窗口限制等")
    else:
        print(f"   ✨ 句子窗口切片在本次测试中表现最好（平均相似度: {avg_window_similarity:.4f}）")
        print(f"   📌 句子窗口切片兼顾检索精确性和上下文完整性，是一种平衡方案。")
        print(f"   📌 适用场景：问答系统、信息检索、需要精确定位但又要提供充足上下文的场景")
        print(f"   📌 特别推荐：当答案可能在单个句子中，但需要周围句子才能完全理解时")
    
    # 提供综合建议
    print(f"\n🎯 综合建议:")
    print(f"   - 如果需要精确检索单个概念或事实 → 推荐句子窗口切片（窗口大小2-3）")
    print(f"   - 如果需要保持段落级别的语义完整性 → 推荐句子切片（块大小512-1024）")
    print(f"   - 如果有严格的 token 数量限制 → 推荐 Token 切片（根据限制调整块大小）")
    print(f"   - 如果文档结构复杂且答案分散 → 建议测试多种方法并结合使用")
    
    print(f"\n{'='*100}")
    print(f"测试完成！")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
