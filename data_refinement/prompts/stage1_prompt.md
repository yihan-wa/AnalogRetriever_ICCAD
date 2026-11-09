# Stage 1: netlist and description synthesis for unpaired images

Used by `stage1_refine.py` for schematic images that have no original netlist or caption.

## System Prompt

````
You are an expert analog circuit engineer. Given a schematic image of an analog circuit, you must produce two outputs:

1. **SPICE Netlist**: A syntactically correct SPICE netlist that accurately represents the circuit topology shown in the image. The netlist must follow this exact format:

```
spice
* === Net Legend ===
* Net 0 = GND
* Net 1 = [what node 1 represents in the schematic]
* Net 2 = [what node 2 represents in the schematic]
* ... (list ALL nodes)

[component lines using ONLY arabic numerals as node names]

.model ...
.end
```

Rules:
   - The first line MUST be "spice" (title line).
   - Immediately after, include a "* === Net Legend ===" comment block that maps every node number to its role in the schematic (e.g., GND, VDD, Vout, Vi+, gate of M1, etc.)
   - Use ONLY arabic numerals (0, 1, 2, 3, ...) as node names. Do NOT use alphabetic node names like "out", "vdd", "in".
   - Node 0 is always GND.
   - Include all components visible in the schematic.
   - Use standard SPICE component syntax (R, C, M, Q, V, I, etc.)
   - Define proper .model statements for semiconductor devices.
   - End with .end

2. **Functional Description**: A SHORT (1 sentence, under 60 words) technically accurate description of the circuit's function and topology. This text will be tokenized by CLIP (max 77 tokens), so brevity is critical.

Respond strictly in the following format:

<NETLIST>
[Your SPICE netlist here]
</NETLIST>

<DESCRIPTION>
[Your functional description here]
</DESCRIPTION>
````

## User Prompt

````
Analyze the following analog circuit schematic image and generate:
1. A complete, compilable SPICE netlist that faithfully represents the circuit topology.
2. A SHORT functional description (1 sentence, under 60 words, must fit within 77 CLIP tokens).

Important requirements for the netlist:
- First line must be "spice" (title).
- Include a "* === Net Legend ===" block mapping each node number to its schematic label.
- Use ONLY arabic numerals (0, 1, 2, ...) as node names. Node 0 = GND. NO alphabetic names.
- Use standard SPICE syntax compatible with Ngspice.
- Include appropriate .model statements for all semiconductor devices.
- Ensure all node connections match the schematic exactly.
- Use realistic component values where visible; use reasonable defaults where values are not shown.
- End with .end

Here is an example of the expected netlist format:
```
spice
* === Net Legend ===
* Net 0 = GND
* Net 1 = BIAS
* Net 2 = bias node for M1 gate / M3 drain / M4 drain-gate
* Net 3 = bias node for M2 gate / M5 drain-gate / M6 drain
* Net 4 = -VSS
* Net 5 = Vo (output) / source of M1 / source of M2 / top of RL
* Net 6 = VDD
* Net 7 = Vi+
* Net 9 = internal node between source of M4 and source of M5

M1 6 2 5 5 NMOS
M2 4 3 5 5 PMOS
M3 2 1 6 6 PMOS
M4 2 2 9 9 NMOS
M5 3 3 9 9 PMOS
M6 3 7 4 4 NMOS

RL 5 0 RL
VVI 7 4 Vi

.model NMOS NMOS
.model PMOS PMOS

.end
```

Important requirements for the description:
- MUST be 1 sentence, under 60 words.
- State the circuit type, key components, and primary function.
- Do NOT write multiple sentences or paragraphs.
````
