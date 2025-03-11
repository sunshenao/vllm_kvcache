import vllm
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer import kv_transfer_agent

# 设置 prompt
prompt = ['''
请围绕“快拍”这一主题，详细阐述其定义、特点、应用场景、优势与劣势，并结合实际案例说明其在现代生活中的重要性。快拍（Snapchat）是一种以“阅后即焚”为核心的社交媒体平台，其独特的瞬时性、娱乐性和互动性吸引了大量用户。请从以下几个方面展开回答：

定义与功能：快拍是什么？它的核心功能有哪些？
用户群体：快拍的主要用户是哪些人？为什么他们喜欢使用快拍？
特点与创新：快拍与传统社交媒体平台相比有哪些独特之处？
应用场景：快拍在日常生活中是如何被使用的？例如，朋友之间的即时分享、品牌营销、新闻传播等。
优势与劣势：快拍的优势是什么？它有哪些潜在的不足或风险？
社会影响：快拍对现代社会的文化、社交方式以及信息传播产生了哪些影响？
技术实现：快拍背后的技术是如何支持其核心功能的？例如，如何实现“阅后即焚”？
未来发展趋势：快拍在未来可能会如何发展？它是否能够持续吸引用户？
请以清晰、逻辑性强的方式进行回答，结合具体数据和案例，以便更好地理解快拍的价值和意义。例如，可以提到快拍在疫情期间如何被用于传播信息，或者它在年轻用户中的流行程度。此外，可以分析快拍在某些领域的成功应用，如品牌如何通过快拍与年轻消费者建立联系，以及快拍在新闻传播中的即时性优势。最后，请总结快拍对社交媒体生态的影响，并对其未来发展提出合理预测。''', 
        #   "你是谁"
          ]*1


# kv_transfer_config = KVTransferConfig(**{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1,"kv_buffer_size":2e9})
# 初始化模型
llm = LLM(model='/root/models/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4') # kv_transfer_config=kv_transfer_config

# 
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=3) #  max_tokens=2

# 进行 prefill
outputs = llm.generate(prompt, sampling_params)
# 0.056 0.041
# 输出 prefill 阶段的结果
for output in outputs:
    print(len(output.outputs[0].text),f"Generated text: {output.outputs[0].text}")

