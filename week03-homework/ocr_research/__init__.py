"""
OCR Research 模块

这个模块提供基于 PaddleOCR 的 LlamaIndex 图像文本加载器。

主要组件:
    - ImageOCRReader: 自定义的图像 OCR 读取器，继承自 LlamaIndex 的 BaseReader
"""

from ocr_research.image_ocr_reader import ImageOCRReader

__all__ = ['ImageOCRReader']
