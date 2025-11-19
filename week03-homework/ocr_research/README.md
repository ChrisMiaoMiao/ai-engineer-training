# ImageOCRReader - 基于 PaddleOCR 的 LlamaIndex 图像文本加载器

这是一个为 LlamaIndex 构建的自定义 OCR 图像文本加载器，使用百度的 PaddleOCR 引擎从图像中提取文本内容。

## 📋 功能特性

- ✅ 继承自 LlamaIndex 的 `BaseReader`，完全兼容 LlamaIndex 生态
- ✅ 支持多种图像格式：PNG, JPG, JPEG, BMP, TIFF, WEBP
- ✅ 支持单个文件或批量文件处理
- ✅ 支持目录批量加载（递归/非递归）
- ✅ 丰富的元数据：置信度、文本块数量、位置信息等
- ✅ 灵活的配置：支持多语言、GPU 加速、自定义参数
- ✅ 详细的代码注释和文档

## 🚀 快速开始

### 1. 环境配置

```bash
# 进入项目目录
cd week03-homework

# 使用 uv 安装依赖（推荐）
uv sync

# 或使用 pip
pip install paddleocr paddlepaddle llama-index llama-index-core
```

### 2. 基本使用

```python
from ocr_research.image_ocr_reader import ImageOCRReader
from llama_index.core import VectorStoreIndex

# 创建 Reader 实例
reader = ImageOCRReader(lang='ch', use_gpu=False)

# 加载单个图像
documents = reader.load_data("image.png")

# 加载多个图像
documents = reader.load_data(["img1.png", "img2.jpg", "img3.png"])

# 从目录批量加载
documents = reader.load_data_from_dir("./images", recursive=True)

# 集成到 LlamaIndex
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("图片中提到了什么内容？")
print(response)
```

### 3. 运行演示

```bash
# 使用 uv run（推荐）
uv run python -m ocr_research.main

# 或激活虚拟环境
source .venv/bin/activate
python -m ocr_research.main
```

## 📚 API 文档

### ImageOCRReader

#### 初始化参数

```python
ImageOCRReader(
    lang='ch',                          # OCR 语言：'ch'(中文), 'en'(英文)等
    use_gpu=False,                      # 是否使用 GPU
    ocr_version='PP-OCRv4',            # PaddleOCR 版本
    use_doc_orientation_classify=False, # 是否启用方向分类
    use_doc_unwarping=False,           # 是否启用图像矫正
    use_textline_orientation=False,    # 是否启用文本行方向分类
    **kwargs                            # 其他 PaddleOCR 参数
)
```

#### 核心方法

**load_data(file, extra_info=None)**
- 从图像文件中提取文本
- `file`: 单个文件路径或文件路径列表
- `extra_info`: 可选的额外元数据字典
- 返回: `List[Document]`

**load_data_from_dir(dir_path, recursive=False, extra_info=None)**
- 从目录批量加载图像
- `dir_path`: 目录路径
- `recursive`: 是否递归搜索子目录
- `extra_info`: 可选的额外元数据
- 返回: `List[Document]`

## 📊 Document 结构

每个 Document 包含以下内容：

### text (文本内容)
```
[Block 1] (conf: 0.98): 第一行文本
[Block 2] (conf: 0.95): 第二行文本
...

=== 纯文本内容 ===
第一行文本
第二行文本
...
```

### metadata (元数据)
- `image_path`: 图像文件绝对路径
- `file_name`: 文件名
- `ocr_model`: OCR 模型版本（如 'PP-OCRv4'）
- `language`: 识别语言
- `num_text_blocks`: 检测到的文本块数量
- `avg_confidence`: 平均识别置信度（0-1）
- `min_confidence`: 最低置信度
- `max_confidence`: 最高置信度
- `text_blocks_detail`: 每个文本块的详细信息
  - `text`: 文本内容
  - `confidence`: 置信度
  - `bbox`: 边界框坐标
  - `block_index`: 块索引
- `used_gpu`: 是否使用了 GPU

## 🔧 高级用法

### 自定义元数据

```python
reader = ImageOCRReader(lang='ch')

# 添加业务相关的元数据
documents = reader.load_data(
    "invoice.png",
    extra_info={
        "document_type": "invoice",
        "source": "scanner",
        "date": "2024-01-01"
    }
)

# 元数据会被合并到 Document.metadata 中
print(documents[0].metadata['document_type'])  # "invoice"
```

### 置信度过滤

```python
# 在处理后过滤低置信度的文本块
for doc in documents:
    high_quality_blocks = [
        block for block in doc.metadata['text_blocks_detail']
        if block['confidence'] > 0.8
    ]
    # 使用高质量的文本块...
```

### 与其他数据源混合

```python
from llama_index.core import SimpleDirectoryReader

# OCR 图像
ocr_reader = ImageOCRReader(lang='ch')
image_docs = ocr_reader.load_data_from_dir("./images")

# 加载 PDF、TXT 等文本文件
text_docs = SimpleDirectoryReader("./documents").load_data()

# 混合索引
all_docs = image_docs + text_docs
index = VectorStoreIndex.from_documents(all_docs)
```

## 📖 使用示例

### 示例 1: 扫描文档处理

```python
reader = ImageOCRReader(lang='ch')
documents = reader.load_data("scanned_document.png")

# 查看识别结果
doc = documents[0]
print(f"识别了 {doc.metadata['num_text_blocks']} 个文本块")
print(f"平均置信度: {doc.metadata['avg_confidence']:.2%}")
print(f"\n文本内容:\n{doc.text}")
```

### 示例 2: 批量处理屏幕截图

```python
reader = ImageOCRReader(lang='ch')

# 处理整个目录
documents = reader.load_data_from_dir(
    "./screenshots",
    recursive=False,
    extra_info={"source": "ui_screenshots"}
)

print(f"处理了 {len(documents)} 个截图")

# 构建索引用于检索
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5)

# 搜索特定内容
response = query_engine.query("找出包含'错误'或'警告'的截图")
print(response)
```

### 示例 3: 多语言 OCR

```python
# 英文 OCR
en_reader = ImageOCRReader(lang='en')
en_docs = en_reader.load_data("english_document.png")

# 中文 OCR
ch_reader = ImageOCRReader(lang='ch')
ch_docs = ch_reader.load_data("chinese_document.png")

# 混合索引
all_docs = en_docs + ch_docs
index = VectorStoreIndex.from_documents(all_docs)
```

## 🎯 测试验证

项目包含了完整的测试脚本 `main.py`，演示了以下功能：

1. **演示 1**: 基本 OCR 功能 - 单个图像处理
2. **演示 2**: 批量处理 - 多个图像处理
3. **演示 3**: LlamaIndex 集成 - 索引构建和查询
4. **演示 4**: 目录加载 - 批量目录处理

运行测试：
```bash
uv run python -m ocr_research.main
```

## 📝 实验报告

完整的实验报告请查看 [report.md](report.md)，包含：

- 架构设计图和数据流程图
- 核心代码详细说明
- OCR 效果评估和错误案例分析
- Document 封装合理性讨论
- 局限性和改进建议
- 技术总结和启发

## 🔍 项目结构

```
ocr_research/
├── __init__.py              # 包初始化
├── image_ocr_reader.py      # ImageOCRReader 核心实现（~400 行）
├── main.py                  # 测试脚本和演示（~350 行）
├── report.md                # 实验报告文档
├── README.md                # 本文件
└── 03.png                   # 测试图像
```

## 💡 最佳实践

1. **选择合适的语言模型**
   - 中文内容使用 `lang='ch'`
   - 英文内容使用 `lang='en'`
   - 混合内容可以分别处理后合并

2. **GPU 加速**
   - 如果有 CUDA 环境，设置 `use_gpu=True` 可以显著提升速度
   - CPU 模式适合小批量处理

3. **质量控制**
   - 检查 `avg_confidence` 评估整体质量
   - 过滤 `confidence < 0.5` 的低质量文本块
   - 对关键内容进行人工校对

4. **性能优化**
   - 批量处理优于逐个处理
   - 考虑使用多进程处理大量图像
   - 预处理图像（去噪、增强）可提高准确率

## ⚠️ 注意事项

1. **PaddleOCR 模型下载**
   - 首次运行会自动下载模型文件（~100MB）
   - 确保网络连接正常

2. **图像质量要求**
   - 推荐分辨率：至少 300 DPI（扫描文档）
   - 避免过度模糊、倾斜的图像
   - 特殊字体可能识别不准确

3. **内存占用**
   - 大图像和批量处理会占用较多内存
   - 建议分批处理大量图像

4. **LlamaIndex 集成**
   - 需要配置 `DASHSCOPE_API_KEY` 才能使用查询功能
   - 纯索引构建不需要 API Key

## 📚 参考资料

- [PaddleOCR 官方文档](https://www.paddleocr.ai/)
- [LlamaIndex 官方文档](https://docs.llamaindex.ai/)
- [LlamaHub Reader 示例](https://llamahub.ai/?tab=readers)

## 📄 许可证

本项目作为教学作业使用，遵循相关课程的许可协议。

---

**作者**: AI Engineer Training - Week 03  
**日期**: 2025-11-19  
**版本**: 1.0.0
