"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from fontTools.ttLib import TTFont
from PIL import Image, ImageFont, ImageDraw
import numpy as np


def get_defined_chars(fontfile):
    ttf = TTFont(fontfile)
    chars = [chr(y) for y in ttf["cmap"].tables[0].cmap.keys()]
    return chars


def get_filtered_chars(fontpath):
    ttf = read_font(fontpath)
    defined_chars = get_defined_chars(fontpath)
    avail_chars = []
    for char in defined_chars:
        try:
            img = np.array(render(ttf, char))
            if img.mean() < 255.:  # Threshold for non-blank characters
                avail_chars.append(char)
        except:
            pass  # Skip characters that cause rendering errors


    return avail_chars


def read_font(fontfile, size=150):
    font = ImageFont.truetype(str(fontfile), size=size)
    return font


def render(font, char, size=(128, 128), pad=20, bottom_pad=20, scale=0.55):
    # Масштабируем размер шрифта
    font_size = int(font.size * scale)
    font = ImageFont.truetype(font.path, font_size)

    # Создаем изображение с достаточным пространством для центрирования текста
    img = Image.new("L", (size[0] + 2*pad, size[1] + pad + bottom_pad), 255)
    draw = ImageDraw.Draw(img)

    # Получаем ограничивающий прямоугольник (bbox) для рисуемого текста
    bbox = draw.textbbox((0, 0), char, font=font)

    # Рассчитываем начальные координаты для центрирования текста
    start_x = (img.width - bbox[2] - bbox[0]) / 2
    start_y = (img.height - bbox[3] - bbox[1]) / 2

    # Рисуем текст по центру изображения
    draw.text((start_x, start_y), char, font=font, fill=0)

    # Обрезаем изображение, оставляя заданный отступ сверху, снизу и по бокам
    img = img.crop((pad, pad, size[0] + pad, size[1] + pad + bottom_pad - pad))
    return img



