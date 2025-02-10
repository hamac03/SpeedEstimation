# ocr_utils.py
from paddleocr import PaddleOCR
import re
from utils import clean_result
from format import format_1_line

def perform_ocr(ocr, image):
    try:
        res = ocr.ocr(image)
        if not res:
            return "No text detected"
        values = [item[1][0] for sublist in res for item in sublist]
        if len(values) == 2:
            result = ''
            for line in values:
                for char in line:
                    if char.isalnum():
                        result += char
                result += '-'
        elif len(values) == 1:
            result = ''
            for line in values:
                for char in line:
                    if char.isalnum():
                        result += char
        else:
            result = '-'.join(values)
        result = result.strip('-')
        print(values)
        print(result)
        return clean_result(format_1_line(result))
    except Exception as e:
        return "No text detected"