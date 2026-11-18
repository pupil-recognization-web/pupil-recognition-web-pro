"""PDF 导出用：不做中英映射，中文原样返回（PDF 已注册中文字体）。"""

def en(text):
    if text is None:
        return ""
    return str(text)
