## Table 2

| Model | Family | MMLU-Pro | HellaSwag | GSM8K | HumanEval | IFEval |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct | diffusion | — | — | 48.0000 | — | — |
| llama3-8b-instruct | autoregressive | — | — | 16.0000 | — | — |

## Table 3

| Model | Family | Wiki F1 | CNN R-L | arXiv BS | Boundary Consistency | Contradiction Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct | diffusion | 0.0000 | — | — | — | — |
| llama3-8b-instruct | autoregressive | 16.3808 | — | — | 4.1667 | 45.8333 |

## Table 4

| Model | Family | Edit Success | Preservation | Over-Edit | Contradiction Reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| llada-8b-instruct / synthetic_repair | diffusion | 84.0000 | 92.9220 | 9.5900 | 24.0000 |
| llama3-8b-instruct / synthetic_repair | autoregressive | 0.0000 | 93.2381 | 94.2166 | 76.0000 |

## Table 5

| Model | Family | GSM8K | MATH | Chain Avg. | WinoG. | Path-Star | Plan Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llada-8b-instruct | diffusion | — | — | — | 0.0000 | — | 0.0000 |
| llama3-8b-instruct | autoregressive | — | — | — | 32.0000 | — | 32.0000 |
