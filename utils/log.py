import string
import random


def generate_random_str(length=30):
    # string.ascii_letters 大小写字母， string.digits 为数字
    characters_long = list(string.ascii_letters + string.digits)

    # 打乱字符串序列
    random.shuffle(characters_long)

    # picking random characters from the list
    password = []
    # 生成密码个数
    for b in range(length):
        password.append(random.choice(characters_long))

        # 打乱密码顺序
        random.shuffle(password)
