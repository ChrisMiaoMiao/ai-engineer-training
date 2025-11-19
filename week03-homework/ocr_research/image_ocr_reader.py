"""
ImageOCRReader: 基于 PaddleOCR 的 LlamaIndex 图像文本加载器

这个模块实现了一个自定义的 LlamaIndex Reader，用于从图像中提取文本内容。
它使用百度的 PaddleOCR 引擎进行光学字符识别（OCR），并将结果转换为
LlamaIndex 可以处理的 Document 对象。

主要功能：
1. 支持单个或批量图像文件处理
2. 提取图像中的文本内容及其位置信息
3. 计算 OCR 识别置信度
4. 生成包含丰富元数据的 Document 对象
"""

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from paddleocr import PaddleOCR
from typing import List, Union, Dict, Any, Optional
import os
from pathlib import Path


class ImageOCRReader(BaseReader):
    """
    使用 PaddleOCR 从图像中提取文本并返回 LlamaIndex Document 对象
    
    这个 Reader 继承自 LlamaIndex 的 BaseReader，实现了从图像文件到
    Document 对象的转换流程。它封装了 PaddleOCR 的功能，使得图像中的
    文本内容可以被整合到 RAG（Retrieval-Augmented Generation）系统中。
    
    Attributes:
        lang (str): OCR 识别语言，默认 'ch' 表示中文
        use_gpu (bool): 是否使用 GPU 加速
        ocr_model (PaddleOCR): PaddleOCR 实例
        ocr_version (str): PaddleOCR 版本
        additional_params (dict): 传递给 PaddleOCR 的额外参数
    """
    
    def __init__(
        self,
        lang: str = 'ch',
        use_gpu: bool = False,
        ocr_version: str = 'PP-OCRv4',
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        **kwargs
    ):
        """
        初始化 ImageOCRReader
        
        Args:
            lang (str): OCR 语言代码
                - 'ch': 中文（默认）
                - 'en': 英文
                - 'fr': 法文等其他语言
            use_gpu (bool): 是否使用 GPU 加速
                - True: 使用 GPU（需要 CUDA 环境）
                - False: 使用 CPU（默认）
            ocr_version (str): PaddleOCR 版本，默认 'PP-OCRv4'
            use_doc_orientation_classify (bool): 是否使用文档方向分类，默认 False
            use_doc_unwarping (bool): 是否使用文本图像矫正，默认 False
            use_textline_orientation (bool): 是否使用文本行方向分类，默认 False
            **kwargs: 其他传递给 PaddleOCR 的参数，如 det_model_dir, rec_model_dir 等
        
        Example:
            >>> reader = ImageOCRReader(lang='ch', use_gpu=True)
            >>> reader = ImageOCRReader(lang='en', ocr_version='PP-OCRv5')
        """
        super().__init__()
        
        self.lang = lang
        self.use_gpu = use_gpu
        self.ocr_version = ocr_version
        self.additional_params = kwargs
        
        # 初始化 PaddleOCR
        # PaddleOCR 是一个轻量级的 OCR 工具，支持检测、识别、方向分类等功能
        # 注意：新版本 PaddleOCR 只接受特定的参数，不支持的参数会报错
        ocr_params = {
            'lang': lang,  # 语言
        }
        
        # 如果使用 GPU，设置使用 GPU
        if use_gpu:
            ocr_params['use_gpu'] = True
        
        # 添加方向分类支持（use_angle_cls 是 PaddleOCR 的标准参数）
        if use_doc_orientation_classify:
            ocr_params['use_angle_cls'] = True
        
        # 合并用户自定义的参数（只添加 PaddleOCR 支持的参数）
        # 过滤掉我们自定义的参数，避免传递给 PaddleOCR
        filtered_kwargs = {
            k: v for k, v in kwargs.items() 
            if k not in ['use_doc_unwarping', 'use_textline_orientation']
        }
        ocr_params.update(filtered_kwargs)
        
        # 创建 PaddleOCR 实例
        self.ocr_model = PaddleOCR(**ocr_params)
        
    def load_data(
        self,
        file: Union[str, Path, List[Union[str, Path]]],
        extra_info: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        从单个或多个图像文件中提取文本，返回 Document 列表
        
        这是 BaseReader 接口要求实现的核心方法。它负责：
        1. 处理输入的文件路径（单个或列表）
        2. 调用 OCR 引擎提取文本
        3. 格式化提取的文本和元数据
        4. 构造并返回 Document 对象列表
        
        Args:
            file (Union[str, Path, List[Union[str, Path]]]): 
                图像文件路径，可以是：
                - 单个文件路径字符串
                - Path 对象
                - 文件路径列表
            extra_info (Optional[Dict[str, Any]]): 
                额外的元数据信息，会被添加到每个 Document 的 metadata 中
        
        Returns:
            List[Document]: LlamaIndex Document 对象列表，每个 Document 包含：
                - text: 从图像中提取的文本内容
                - metadata: 包含图像路径、OCR 信息、置信度等元数据
        
        Raises:
            FileNotFoundError: 当指定的图像文件不存在时
            ValueError: 当文件格式不支持时
        
        Example:
            >>> reader = ImageOCRReader()
            >>> # 单个文件
            >>> docs = reader.load_data("image.png")
            >>> # 多个文件
            >>> docs = reader.load_data(["img1.png", "img2.jpg"])
            >>> # 带额外元数据
            >>> docs = reader.load_data("image.png", extra_info={"source": "scanner"})
        """
        # 统一处理输入：将单个路径转换为列表
        if isinstance(file, (str, Path)):
            files = [file]
        else:
            files = file
        
        # 存储所有 Document 对象
        documents = []
        
        # 遍历每个图像文件
        for file_path in files:
            # 转换为 Path 对象，方便路径操作
            file_path = Path(file_path)
            
            # 验证文件是否存在
            if not file_path.exists():
                raise FileNotFoundError(f"图像文件不存在: {file_path}")
            
            # 验证文件格式（支持常见图像格式）
            supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
            if file_path.suffix.lower() not in supported_formats:
                raise ValueError(
                    f"不支持的文件格式: {file_path.suffix}. "
                    f"支持的格式: {', '.join(supported_formats)}"
                )
            
            # 执行 OCR 识别
            # PaddleOCR 使用 ocr() 方法进行识别，返回包含检测框和识别结果的列表
            result = self.ocr_model.ocr(str(file_path))
            
            # 提取文本内容和元数据
            text_content, metadata = self._process_ocr_result(
                result, 
                file_path,
                extra_info
            )
            
            # 创建 Document 对象
            # Document 是 LlamaIndex 的核心数据结构，包含文本和元数据
            document = Document(
                text=text_content,
                metadata=metadata
            )
            
            documents.append(document)
        
        return documents
    
    def _process_ocr_result(
        self,
        ocr_result: Any,
        file_path: Path,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> tuple[str, Dict[str, Any]]:
        """
        处理 PaddleOCR 的识别结果，提取文本和元数据
        
        PaddleOCR 返回的结果是一个复杂的数据结构，包含：
        - 文本检测框的坐标
        - 识别出的文本内容
        - 识别置信度
        
        这个方法负责解析这些信息，并格式化为易于使用的形式。
        
        Args:
            ocr_result: PaddleOCR 的识别结果对象
            file_path (Path): 图像文件路径
            extra_info (Optional[Dict[str, Any]]): 用户提供的额外元数据
        
        Returns:
            tuple[str, Dict[str, Any]]: 
                - str: 格式化后的文本内容
                - dict: 包含详细元数据的字典
        """
        # 存储所有识别出的文本块
        text_blocks = []
        # 存储每个文本块的置信度
        confidences = []
        # 存储文本块的详细信息（位置、内容、置信度）
        detailed_blocks = []
        
        # 遍历 OCR 结果
        # PaddleOCR 的返回格式可能因版本而异
        # 新版本 (PaddleX): 返回 OCRResult 对象
        # 旧版本: 返回 [[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)], ...]
        
        # 首先检查返回结果的类型
        if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0:
            result_item = ocr_result[0]
            
            # 新版本 PaddleX 返回 OCRResult 对象（类字典对象）
            if hasattr(result_item, 'keys') and callable(result_item.keys):
                # OCRResult 是类字典对象
                # 提取文本和相关信息
                # 常见的键: 'dt_polys', 'rec_texts', 'rec_scores'
                if 'rec_texts' in result_item:
                    rec_texts = result_item.get('rec_texts', [])
                    rec_scores = result_item.get('rec_scores', [])
                    dt_polys = result_item.get('dt_polys', [])
                    
                    for idx, text in enumerate(rec_texts):
                        confidence = rec_scores[idx] if idx < len(rec_scores) else 1.0
                        bbox = dt_polys[idx] if idx < len(dt_polys) else None
                        
                        text_blocks.append(text)
                        confidences.append(float(confidence))
                        detailed_blocks.append({
                            'text': text,
                            'confidence': float(confidence),
                            'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                            'block_index': idx
                        })
                else:
                    print(f"警告: OCRResult 中未找到 'rec_texts' 键")
            
            # 旧版本格式: 直接返回字符串列表
            elif isinstance(result_item, str):
                for idx, text in enumerate(ocr_result):
                    if text:
                        text_blocks.append(text)
                        confidences.append(1.0)
                        detailed_blocks.append({
                            'text': text,
                            'confidence': 1.0,
                            'bbox': None,
                            'block_index': idx
                        })
            
            # 旧版本格式: 嵌套列表
            elif isinstance(result_item, list):
                for idx, line in enumerate(result_item):
                    if line and len(line) >= 2:
                        box = line[0]
                        
                        if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                            text = line[1][0]
                            confidence = line[1][1]
                        else:
                            print(f"警告: 第 {idx} 行的格式不符合预期: {line}")
                            continue
                        
                        text_blocks.append(text)
                        confidences.append(float(confidence))
                        detailed_blocks.append({
                            'text': text,
                            'confidence': float(confidence),
                            'bbox': box,
                            'block_index': idx
                        })
        
        # 格式化文本内容
        # 每个文本块单独一行，并附带置信度信息
        formatted_text = self._format_text_blocks(text_blocks, confidences)
        
        # 计算平均置信度
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        
        # 构建元数据字典
        # 注意：为了避免 LlamaIndex 的 metadata 长度限制，我们简化元数据
        metadata = {
            # 原始图像路径
            'image_path': str(file_path.absolute()),
            # 图像文件名
            'file_name': file_path.name,
            # 使用的 OCR 模型版本
            'ocr_model': self.ocr_version,
            # OCR 语言
            'language': self.lang,
            # 检测到的文本块数量
            'num_text_blocks': len(text_blocks),
            # 平均识别置信度
            'avg_confidence': round(avg_confidence, 4),
            # 最低置信度（用于质量评估）
            'min_confidence': round(min(confidences), 4) if confidences else 0.0,
            # 最高置信度
            'max_confidence': round(max(confidences), 4) if confidences else 0.0,
            # 是否使用了 GPU
            'used_gpu': self.use_gpu,
            # 注意：text_blocks_detail 包含大量数据，可能导致 LlamaIndex 元数据过长
            # 如果需要详细信息，可以单独存储或通过其他方式访问
            # 'text_blocks_detail': detailed_blocks,  # 默认注释掉
        }
        
        # 合并用户提供的额外元数据
        if extra_info:
            metadata.update(extra_info)
        
        return formatted_text, metadata
    
    def _format_text_blocks(
        self,
        text_blocks: List[str],
        confidences: List[float]
    ) -> str:
        """
        格式化文本块，生成易读的文本内容
        
        将 OCR 识别出的多个文本块组织成结构化的文本，
        便于后续的检索和理解。
        
        Args:
            text_blocks (List[str]): 识别出的文本块列表
            confidences (List[float]): 对应的置信度列表
        
        Returns:
            str: 格式化后的文本字符串
        
        格式示例：
            [Block 1] (conf: 0.98): 这是第一行文本
            [Block 2] (conf: 0.95): 这是第二行文本
            
            === 纯文本内容 ===
            这是第一行文本
            这是第二行文本
        """
        if not text_blocks:
            return ""
        
        # 方式1: 带置信度的详细格式
        detailed_lines = []
        for i, (text, conf) in enumerate(zip(text_blocks, confidences), 1):
            # 格式: [Block N] (conf: 0.XX): 文本内容
            detailed_lines.append(f"[Block {i}] (conf: {conf:.2f}): {text}")
        
        detailed_text = "\n".join(detailed_lines)
        
        # 方式2: 纯文本格式（用于简单场景）
        plain_text = "\n".join(text_blocks)
        
        # 组合两种格式
        # 这样既保留了详细信息，又提供了纯文本版本
        formatted_text = f"{detailed_text}\n\n=== 纯文本内容 ===\n{plain_text}"
        
        return formatted_text
    
    def load_data_from_dir(
        self,
        dir_path: Union[str, Path],
        recursive: bool = False,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        从目录批量加载图像文件（附加功能）
        
        这是一个便捷方法，用于处理包含多个图像的目录。
        它会自动发现目录中的所有图像文件并进行 OCR 处理。
        
        Args:
            dir_path (Union[str, Path]): 目录路径
            recursive (bool): 是否递归搜索子目录，默认 False
            extra_info (Optional[Dict[str, Any]]): 额外元数据
        
        Returns:
            List[Document]: Document 对象列表
        
        Example:
            >>> reader = ImageOCRReader()
            >>> # 加载单个目录
            >>> docs = reader.load_data_from_dir("./images")
            >>> # 递归加载所有子目录
            >>> docs = reader.load_data_from_dir("./all_images", recursive=True)
        """
        dir_path = Path(dir_path)
        
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"目录不存在或不是有效目录: {dir_path}")
        
        # 支持的图像格式
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
        
        # 查找所有图像文件
        if recursive:
            # 递归查找
            image_files = [
                f for f in dir_path.rglob('*')
                if f.suffix.lower() in image_extensions and f.is_file()
            ]
        else:
            # 只查找当前目录
            image_files = [
                f for f in dir_path.glob('*')
                if f.suffix.lower() in image_extensions and f.is_file()
            ]
        
        if not image_files:
            print(f"警告: 在 {dir_path} 中未找到图像文件")
            return []
        
        print(f"找到 {len(image_files)} 个图像文件，开始处理...")
        
        # 批量处理
        return self.load_data(image_files, extra_info=extra_info)
