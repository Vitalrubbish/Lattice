# A Runtime-Level Fragmentation Calculator

**2026-5-30** written by **Vitalrubbish**

## Background

Now your calculation of fragmentation rate is based on single request, use the block-size and VMM minimum size and prompt length to calculate the fragmentation ratio statically. However, this method is far from accurate since in real inference senarios, LLM serves muti-requests at a time. This means that statically computing framentation ratio is infeasible.

## Computing Mechanism

In this part you need to implement a runtime-level fragmentation ratio calculator. The LLM receives multi-requests, which vary in prompt length. In each step, you need to calculate the fragmentation ratio by the following mechanism:

1. You need to compute the size of memory allocated but not in the free-list.
2. You need to compute the size of memory which is currently occupied by active tokens.
3. And compute the ratio by 1 - (memory occupied by active tokens) / (memory allocated but not freed).
4. Please calculate the average number of fragmentation ratio by time, and get the final answer.

## Tests

1. Please find a prompt dataset, and get a small sample of this dataset to count the prompt length distribution in real inference senarios.
2. And construct the test bench according to this distribution. Then add this test into step3-test-wsl2.sh.
