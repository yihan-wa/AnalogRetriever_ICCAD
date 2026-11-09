# Stage 2b: description refinement (paper Fig. 5)

Used by `stage2_refine.py` after a failed netlist has been repaired and verified by Ngspice.

## System Prompt

````
You are an expert analog circuit engineer. Your task is to produce a technically accurate functional description of an analog circuit. You will be given:
1. A corrected, compilation-verified SPICE netlist.
2. The original (possibly inaccurate) caption/description.

Using the verified netlist as the ground truth, produce a refined description that is technically precise and faithful to the actual circuit topology.
````

## User Prompt

````
Given the following verified SPICE netlist and its original description, produce a refined, technically accurate functional description of the circuit.

**Verified SPICE Netlist:**
```spice
{netlist}
```

**Original Description:**
{original_description}

Requirements:
- MUST be 1 sentence, under 60 words (will be tokenized by CLIP, max 77 tokens).
- Name the circuit type/category (e.g., amplifier, voltage reference, oscillator).
- List the key components and their roles.
- Describe the primary operating principle.
- Do NOT write multiple sentences or paragraphs.

Return ONLY the refined description in the following format:

<DESCRIPTION>
[Refined functional description]
</DESCRIPTION>
````
