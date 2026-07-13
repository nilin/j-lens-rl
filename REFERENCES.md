# References and credits

The following external projects materially informed or support this repository.

## Jacobian Lens

- Repository: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
- Paper: [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
- Use here: the `jlens` package is installed directly from commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` and provides lens fitting, storage, and Jacobian transport.

## TRL

- Repository: [huggingface/trl](https://github.com/huggingface/trl)
- Use here: TRL's GRPO trainer and accuracy-reward conventions were used as an algorithmic reference for the local group-relative policy-gradient loop. TRL code is not vendored, installed, or imported by this repository.

## Qwen2.5

- Repository: [QwenLM/Qwen2.5](https://github.com/QwenLM/Qwen2.5)
- Model: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- Use here: base policy, frozen reference policy, tokenizer, and model unembedding used by the J-lens readout.

## GSM8K

- Repository: [openai/grade-school-math](https://github.com/openai/grade-school-math)
- Dataset: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
- Use here: RL prompts, verifiable numeric rewards, and held-out evaluation.

## WikiText

- Dataset: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)
- Original project: [Salesforce Research WikiText](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/)
- Use here: generic text used to fit and calibrate the Jacobian lens, separate from GSM8K answers.
