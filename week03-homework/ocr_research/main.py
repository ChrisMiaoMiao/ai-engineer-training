"""
OCR 图像文本加载器测试脚本

这个脚本演示了如何使用 ImageOCRReader 从图像中提取文本，
并将其集成到 LlamaIndex 进行检索增强生成（RAG）。

运行方式:
    python -m ocr_research.main
    或
    uv run python -m ocr_research.main
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 导入 LlamaIndex 核心组件
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels
)

# 导入我们实现的 ImageOCRReader
from ocr_research.image_ocr_reader import ImageOCRReader


def setup_llm():
    """
    配置 LlamaIndex 使用的大语言模型和嵌入模型
    
    这里使用阿里云的 DashScope API，兼容 OpenAI 接口。
    需要设置环境变量 DASHSCOPE_API_KEY
    """
    # 加载环境变量
    load_dotenv()
    
    # 检查 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("警告: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("如需使用 LlamaIndex 查询功能，请设置此环境变量")
        return False
    
    # 配置大语言模型（用于生成回答）
    Settings.llm = OpenAILike(
        model="qwen-plus",  # 使用通义千问 Plus 模型
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        is_chat_model=True
    )
    
    # 配置嵌入模型（用于文本向量化和检索）
    Settings.embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        embed_batch_size=6,
        embed_input_length=8192
    )
    
    print("✓ LLM 和嵌入模型配置成功")
    return True


def demo_basic_ocr():
    """
    演示 1: 基本的 OCR 功能
    
    展示如何使用 ImageOCRReader 从单个图像中提取文本
    """
    print("\n" + "="*60)
    print("演示 1: 基本 OCR 功能 - 多图像测试")
    print("="*60)
    
    # 创建 ImageOCRReader 实例
    # lang='ch' 表示使用中文模型
    reader = ImageOCRReader(lang='ch', use_gpu=False)
    
    # 测试图像列表
    test_images = [
        "paddle-1.png",
        "paddle-2.png", 
        "paddle-3.jpg"
    ]
    
    all_documents = []
    
    for img_name in test_images:
        image_path = Path(__file__).parent / img_name
        
        if not image_path.exists():
            print(f"警告: 测试图像不存在 - {image_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"正在处理图像: {image_path.name}")
        print(f"{'='*60}")
        
        # 加载图像并进行 OCR
        documents = reader.load_data(str(image_path))
        
        # 检查结果
        if not documents:
            print("错误: 未能从图像中提取文本")
            continue
        
        # 获取第一个 Document
        doc = documents[0]
        all_documents.extend(documents)
        
        # 打印提取的文本（限制长度）
        print("\n--- 提取的文本内容（前200字符）---")
        text_preview = doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
        print(text_preview)
        
        # 打印元数据
        print("\n--- 元数据信息 ---")
        metadata = doc.metadata
        print(f"图像文件: {metadata['file_name']}")
        print(f"OCR 模型: {metadata['ocr_model']}")
        print(f"语言: {metadata['language']}")
        print(f"文本块数量: {metadata['num_text_blocks']}")
        print(f"平均置信度: {metadata['avg_confidence']:.2%}")
        print(f"置信度范围: {metadata['min_confidence']:.2%} - {metadata['max_confidence']:.2%}")
    
    print(f"\n{'='*60}")
    print(f"总结: 成功处理 {len(all_documents)} 个图像")
    print(f"{'='*60}")
    
    return all_documents


def demo_batch_ocr():
    """
    演示 2: 批量处理图像（如果有多个图像）
    
    展示如何一次处理多个图像文件
    """
    print("\n" + "="*60)
    print("演示 2: 批量 OCR 处理")
    print("="*60)
    
    # 获取当前目录下所有图像文件
    current_dir = Path(__file__).parent
    image_files = list(current_dir.glob("*.png")) + list(current_dir.glob("*.jpg"))
    
    if not image_files:
        print("提示: 当前目录没有找到其他图像文件")
        return None
    
    print(f"\n找到 {len(image_files)} 个图像文件:")
    for img in image_files:
        print(f"  - {img.name}")
    
    # 创建 Reader
    reader = ImageOCRReader(lang='ch')
    
    # 批量处理
    print("\n开始批量处理...")
    documents = reader.load_data([str(f) for f in image_files])
    
    print(f"\n成功处理 {len(documents)} 个图像")
    
    # 显示摘要统计
    total_blocks = sum(doc.metadata['num_text_blocks'] for doc in documents)
    avg_confidence = sum(doc.metadata['avg_confidence'] for doc in documents) / len(documents)
    
    print(f"总文本块数: {total_blocks}")
    print(f"总体平均置信度: {avg_confidence:.2%}")
    
    return documents


def demo_llamaindex_integration(documents):
    """
    演示 3: 集成到 LlamaIndex 进行检索查询
    
    展示如何将 OCR 提取的文本构建索引，并进行智能检索
    
    Args:
        documents: ImageOCRReader 生成的 Document 列表
    """
    print("\n" + "="*60)
    print("演示 3: LlamaIndex 集成 - 构建索引并查询")
    print("="*60)
    
    if not documents:
        print("错误: 没有可用的文档")
        return
    
    # 配置 LLM（如果还没配置）
    llm_configured = setup_llm()
    
    if not llm_configured:
        print("\n跳过 LlamaIndex 查询演示（需要配置 DASHSCOPE_API_KEY）")
        print("但是索引构建功能仍然可用...")
    
    try:
        # 步骤 1: 构建向量索引
        print("\n正在构建向量索引...")
        
        # 配置更大的 chunk size 以容纳 OCR 元数据
        # OCR 提取的文本通常比较长，需要更大的分块大小
        from llama_index.core.node_parser import SentenceSplitter
        
        # 设置文本分割器，增加 chunk_size 以避免元数据长度超限
        Settings.text_splitter = SentenceSplitter(
            chunk_size=2048,  # 增加到 2048（原默认值 1024）
            chunk_overlap=200  # 保持合理的重叠
        )
        
        index = VectorStoreIndex.from_documents(documents)
        print("✓ 索引构建完成")
        
        # 如果 LLM 已配置，进行查询演示
        if llm_configured:
            # 步骤 2: 创建查询引擎
            query_engine = index.as_query_engine(
                similarity_top_k=3,  # 返回最相似的3个文本块
                streaming=False
            )
            
            # 步骤 3: 执行查询
            print("\n--- 查询测试 ---")
            
            # 查询示例（根据实际图像内容调整）
            queries = [
                "这张图片中提到了什么内容？",
                "图片中有哪些关键信息？",
                "请总结图片中的文字内容",
            ]
            
            for i, query in enumerate(queries[:1], 1):  # 只执行第一个查询作为示例
                print(f"\n查询 {i}: {query}")
                print("-" * 40)
                
                try:
                    response = query_engine.query(query)
                    print(f"回答: {response}")
                    
                    # 显示检索到的源文本
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        print(f"\n检索到 {len(response.source_nodes)} 个相关文本块:")
                        for j, node in enumerate(response.source_nodes[:2], 1):
                            print(f"\n  文本块 {j} (相似度: {node.score:.3f}):")
                            print(f"  {node.text[:200]}...")
                            if 'image_path' in node.metadata:
                                print(f"  来源: {Path(node.metadata['image_path']).name}")
                
                except Exception as e:
                    print(f"查询出错: {e}")
        
        else:
            print("\n提示: 要体验完整的查询功能，请设置 DASHSCOPE_API_KEY")
            print("索引已成功构建，可以用于其他检索场景")
    
    except Exception as e:
        print(f"索引构建或查询出错: {e}")
        import traceback
        traceback.print_exc()


def demo_directory_loading():
    """
    演示 4: 从目录加载图像（附加功能）
    
    展示便捷的目录批量加载功能
    """
    print("\n" + "="*60)
    print("演示 4: 从目录批量加载")
    print("="*60)
    
    current_dir = Path(__file__).parent
    
    # 创建 Reader
    reader = ImageOCRReader(lang='ch')
    
    try:
        # 从当前目录加载所有图像
        print(f"\n从目录加载: {current_dir}")
        documents = reader.load_data_from_dir(
            current_dir,
            recursive=False,  # 不递归子目录
            extra_info={"source": "ocr_research", "batch": "demo"}
        )
        
        if documents:
            print(f"✓ 成功加载 {len(documents)} 个文档")
            
            # 显示每个文档的简要信息
            for i, doc in enumerate(documents, 1):
                print(f"\n文档 {i}:")
                print(f"  文件: {doc.metadata['file_name']}")
                print(f"  文本块: {doc.metadata['num_text_blocks']}")
                print(f"  置信度: {doc.metadata['avg_confidence']:.2%}")
                print(f"  额外信息: source={doc.metadata.get('source')}, batch={doc.metadata.get('batch')}")
        
        return documents
    
    except Exception as e:
        print(f"目录加载失败: {e}")
        return None


def main():
    """
    作业的入口函数
    
    依次运行各个演示，展示 ImageOCRReader 的完整功能
    """
    print("="*60)
    print("ImageOCRReader 功能演示")
    print("基于 PaddleOCR 的 LlamaIndex 图像文本加载器")
    print("="*60)
    
    # 演示 1: 基本 OCR
    documents = demo_basic_ocr()
    
    # 演示 2: 批量处理
    # batch_documents = demo_batch_ocr()
 
    # 演示 3: LlamaIndex 集成
    if documents:
        demo_llamaindex_integration(documents)
    
    # 演示 4: 目录加载
    # dir_documents = demo_directory_loading()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    print("\n提示:")
    print("1. 可以取消注释其他演示函数来查看更多功能")
    print("2. 将更多图像文件放入 ocr_research 目录进行测试")
    print("3. 设置 DASHSCOPE_API_KEY 环境变量以启用 LlamaIndex 查询功能")


if __name__ == "__main__":
    main()