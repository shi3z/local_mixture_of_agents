"""
Local Mixture of Agents  

based on https://github.com/togethercomputer/MoA/tree/main

          Code by shi3z(2024/7/5)
"""
import datasets
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
import openai
import time
import requests
from functools import partial
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
import copy

models=[
    ["lightblue/karasu-7B",5001],
    ["karakuri-ai/karakuri-lm-8x7b-chat-v0.1",5002],
    ["elyza/Llama-3-ELYZA-JP-8B",5004]
]

llm=""

def generate_llm(
    messages,
    model="dummy",
    max_tokens=2048,
    temperature=0.7,):

    output=llm(messages)

    return output.text



def generate_vllm(
    messages,
    model=["lightblue/karasu-7B",5001],
    max_tokens=2048,
    temperature=0.7,
):

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        #try:
            modelpath,port=model

            endpoint = f"http://localhost:{port}/v1/chat/completions"


            res = requests.post(
                endpoint,
                json={
                    "model": modelpath,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
            )
            #print(res)
            output = res.json()["choices"][0]["message"]["content"]

            break

        #except Exception as e:
        #    print(e)
        #    time.sleep(sleep_time)

    if output is None:

        return output

    output = output.strip()

    return output


welcome_message = """
# Welcome to the Free AI MoA (Mixture-of-Agents)  demo!

Mixture of Agents (MoA) is a novel approach that leverages the collective strengths of multiple LLMs to enhance performance, achieving state-of-the-art results. By employing a layered architecture where each layer comprises several LLM agents, MoA significantly outperforms GPT-4 Omni’s 57.5% on AlpacaEval 2.0 with a score of 65.1%, using only open-source models!

This demo uses the following LLMs as reference models, then passes the results to the aggregate model for the final response:
- internlm/internlm2_5-7b-chat-1m
"""

for model in models:
     welcome_message = f"{welcome_message}\n- {model[0]}"

def loadHugeLLM(session_len=1048576,tp=4):
    backend_config = TurbomindEngineConfig(
        rope_scaling_factor=2.5,
        session_len=104857, #6,  # 1M context length
        max_batch_size=1,
        cache_max_entry_count=0.7,
        tp=1) #XY4)  # 4xA100-80G.
    pipe = pipeline('internlm/internlm2_5-7b-chat-1m', backend_config=backend_config)
    return pipe 



def inject_references_to_messages(
    messages,
    references,
):

    messages = copy.deepcopy(messages)

    system = f"""最新のユーザー クエリに対するさまざまなオープンソース モデルからの一連の応答が提供されています。あなたの仕事は、これらの応答を 1 つの高品質な応答に合成することです。これらの回答で提供された情報には偏りや不正確なものがある可能性があることを認識し、批判的に評価することが重要です。回答は、与えられた回答を単に再現するだけでなく、指示に対する洗練された正確かつ包括的な回答を提供する必要があります。応答が適切に構成され、一貫性があり、最高の精度と信頼性の基準に準拠していることを確認してください。

モデルからの応答:"""

    for i, reference in enumerate(references):

        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":

        messages[0]["content"] += "\n\n" + system

    else:

        messages = [{"role": "system", "content": system}] + messages

    return messages


def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=2048,
    temperature=0.7,
    llm=generate_vllm
    ):

    if len(references) > 0:

        messages = inject_references_to_messages(messages, references)

    return llm(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

def process_fn(
    item,
    temperature=0.7,
    max_tokens=2048,
):
    references = item.get("references", [])
    model = item["model"]
    messages = item["instruction"]

    output = generate_with_references(
        model=model,
        messages=messages,
        references=references,
        temperature=temperature,
        max_tokens=max_tokens,
        llm=generate_vllm
    )
    print(f"\nFinished querying [bold]{model}.[/bold]")

    return {"output": output}
console = Console()

def main(temperature=0.7,rounds=3,max_tokens=2048):
    global llm
    llm = loadHugeLLM(session_len=104857,tp=1)
    instruction= "何か面白いこと言って."

    md = Markdown(welcome_message)    

    data = {
        "instruction": [[ {"role": "system", "content": "あなたはイカしたアシスタントです"}] 
                                for _ in range(len(models))],
        "references": [""] * len(models),
        "model": [m for m in models],
    }

    print(data)
    for i in range(len(models)):
        data["instruction"][i].append({"role": "user", "content": instruction})
        data["references"] = [""] * len(models)
    print(data)
    #eval_set = datasets.Dataset.from_dict(data)
    num_proc=len(models)


    for i_round in range(rounds):
        references = data["references"]
        for i in range(num_proc):
            output = process_fn({"model":data["model"][i],
                                "references":data["references"][i],
                                "instruction":data["instruction"][i]})
            references[i] = output["output"]
        data["references"] = references
        print(data)
    

    output = generate_with_references(
        model=model,
        messages=messages,
        references=references,
        temperature=temperature,
        max_tokens=8192,
        llm=generate_llm
    )
    print(output)

    
    
if __name__ =="__main__":
    main()
