# Local LLM Mixture of Agents (MoA) 

## Overview

Our company's president, Tsugunosuke (an AI supercomputer), has developed a Mixture of Agents (MoA) that combines Japanese Language Models (LLMs) using the MoA technique.

MoA is a method that can surpass the performance of proprietary LLMs like GPT-4o using only a combination of open-source LLM models. It has been shown to exceed GPT-4o's performance in English.

For more information on the original MoA project, visit: [Together Computer's MoA GitHub Repository](https://github.com/togethercomputer/MoA/)

This project is a research outcome of FreeAI Corporation. [FreeAI Ltd.](https://free-ai.ltd)

## Our Approach

While the sample code in the original MoA project relies on Together's API, we've developed a fully local MoA solution to accommodate proprietary data. Our implementation runs on Tsugunosuke and combines the following LLMs using the MoA technique:

- internlm/internlm2_5-7b-chat-1m
- lightblue/karasu-7B
- karakuri-ai/karakuri-lm-8x7b-chat-v0.1
- elyza/Llama-3-ELYZA-JP-8B

We use karasu-7B, Karakuri-LM-8x7B, and Llama-3-ELYZA-JP-8B as reference models, while InternLM2.5-7B-Chat-1M (capable of handling 1 million token lengths) serves as the conclusion-drawing model.

## Technical Setup

Our setup utilizes Tsugunosuke's eight A100 80GB GPUs:

- Three models (karasu-7B, Karakuri-LM-8x7B, and Llama-3-ELYZA-JP-8B) are launched using vllm.
- InternLM2.5-7B-chat-1M, which is not compatible with vllm, is launched separately using Lmdeploy.

This configuration allows us to construct our MoA system entirely locally.

## Features

- Fully local implementation of MoA
- Combines the strengths of multiple Japanese LLMs
- Capable of handling 1 million token lengths
- Designed to work with proprietary data

## License

MIT

## Acknowledgements

This project is inspired by and builds upon the work done by Together Computer on their MoA project. We extend our gratitude to their team for their groundbreaking research in this area.
