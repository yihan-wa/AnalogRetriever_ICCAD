# Dataset refinement (paper Sec. 4.1, Fig. 4–5, Table 1)

Two cascaded LLM stages repair Masala-CHAI, with Ngspice as the
ground-truth oracle between them. All LLM calls use GPT-5.4 through
OpenRouter (`openai/gpt-5.4`, temperature 0.3).

1. **Raw check.** Every original netlist is compiled with Ngspice and run
   through a DC operating-point (`.op`) check.
2. **Stage 1: initial netlist repair.** Each of the 6,069 original netlists is
   sent to the LLM together with its Ngspice results and the schematic
   ([`prompts/stage1_repair_prompt.md`](prompts/stage1_repair_prompt.md)); the
   original caption is kept. For the 302 schematics without a netlist or
   caption, the LLM synthesises both from the image
   ([`prompts/stage1_prompt.md`](prompts/stage1_prompt.md)).
3. **Stage 2: feedback-guided refinement.** Triplets that pass both checks are
   kept. A netlist that still fails is repaired with the Ngspice error log
   ([`prompts/stage2_netlist_repair_prompt.md`](prompts/stage2_netlist_repair_prompt.md), Fig. 5 Stage 2a),
   re-checked and resent with updated feedback for up to 5 iterations. Once it
   passes, its caption is rewritten from the verified netlist and the
   original caption
   ([`prompts/stage2_description_refinement_prompt.md`](prompts/stage2_description_refinement_prompt.md), Fig. 5 Stage 2b).
4. **Table 1.** The checker results of the three datasets are aggregated by
   [`generate_table1.py`](generate_table1.py).

## Usage

Inputs are the original Masala-CHAI release with `images/`, its netlists and
captions, and a `data_mapping.json` listing `image`, `spice` and `caption`
paths per circuit. Ngspice and an OpenRouter API key are required.

```bash
export DATA_DIR=/path/to/raw/Masala-CHAI
export OPENROUTER_API_KEY=...
# Check the inputs without calling the API (expects 6,069 mapped and 302 unmapped images)
python3 data_refinement/stage1_refine.py --data-dir "$DATA_DIR" --mapping "$DATA_DIR/data_mapping.json" \
    --validate-only --expected-original 6069 --expected-generated 302
bash data_refinement/run_refinement.sh            # optional START END arguments process a slice
```

Outputs are written to `data_refinement/output/` (or `$REFINEMENT_RUN_DIR`):
`stage1/` and `stage2/` hold the refined triplets with their
`data_mapping.json`, and `results/` holds the checker outputs and
`table1.json`/`table1.txt`. Both stages record per-sample progress and can be
resumed; a sample is reprocessed, with new API requests, when its inputs,
prompts or the pipeline code change.

| File | Role |
|---|---|
| [`run_refinement.sh`](run_refinement.sh) | Runs the raw check, both stages, the checks after each stage and Table 1. |
| [`stage1_refine.py`](stage1_refine.py) | Stage 1 repair of original netlists and synthesis of missing pairs. |
| [`stage2_refine.py`](stage2_refine.py) | Stage 2 iterative netlist repair and description refinement. |
| [`spice_checker.py`](spice_checker.py) | Ngspice compilation and DC operating-point checks. |
| [`llm_client.py`](llm_client.py) | OpenRouter chat-completion client with image input. |
| [`pipeline_io.py`](pipeline_io.py) | Prompt loading, progress files and output fingerprints. |
| [`generate_table1.py`](generate_table1.py) | Table 1 from the checker outputs. |

## Released dataset

[`results/table1.txt`](results/table1.txt) records the checks of the run that
produced the released dataset
([`eulermaxwell/AnalogRetriever-data`](https://huggingface.co/datasets/eulermaxwell/AnalogRetriever-data),
revision `0c053b6`), which gives the numbers of paper Table 1. That run used
an earlier implementation of this pipeline, which differed from the
procedure above in two respects: Stage 1 generated the netlist and
description of every schematic from the image, with the original netlist
recorded but not given to the model; and Stage 2 processed all triplets,
rewriting the description of every compiling netlist from its Stage-1
description. The code in this directory implements the procedure described
in the paper and has not been used to regenerate the released data.

The raw archive we examined maps 6,069 images, 32 of which point to missing
image files; `--validate-only` lists them so that the mapping can be
reconciled before a full run.
