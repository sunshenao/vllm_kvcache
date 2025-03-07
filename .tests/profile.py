import json
import numpy as np

# 打开并读取 JSON 文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 使用示例
file_path = '/root/code/latest/Mooncake/vllm/.tests/disagg_prefill-qps-8.json'
data = read_json_file(file_path)


itl = np.array(data['itls'])
print("全部的mean 和 media:",np.mean(itl),np.median(itl))

print("前100的mean 和 media:",np.mean(itl[:100]),np.median(itl[:100]))


print("前125的mean 和 media:",np.mean(itl[:125]),np.median(itl[:125]))


