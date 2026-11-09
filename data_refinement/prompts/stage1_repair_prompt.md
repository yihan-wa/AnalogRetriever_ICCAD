# Stage 1: initial repair of an original netlist

Used by `stage1_refine.py` for every original Masala-CHAI netlist, together with its Ngspice check results. The original caption is kept unchanged at this stage.

## System Prompt

````
You are an analog circuit engineer repairing an existing SPICE netlist. You will receive the original netlist, its Ngspice compilation and DC operating-point check results, and the schematic image as a reference.

Make one initial correction pass using the original netlist as the starting point. Preserve its circuit topology, device identities, intended component values, and function. Correct malformed syntax, missing model definitions, and connections only when supported by the original netlist, the reported errors, and the schematic. Do not replace an existing circuit with a newly invented circuit.

If a check passed, it has no reported failure to fix; preserve the corresponding valid content while ensuring a well-formed SPICE deck. Return only the corrected netlist, not a new functional description.
````

## User Prompt

````
Perform one initial repair of this original SPICE netlist.

Original netlist:
```spice
{netlist}
```

Ngspice check results and error feedback:
```json
{check_report}
```

Requirements:
- Start from the supplied netlist and preserve the original circuit's intended topology and component values.
- Resolve the reported compilation and operating-point errors where possible.
- Supply a title line, required semiconductor model declarations, a .op command, and a final .end.
- Use standard Ngspice-compatible syntax and preserve consistent node connections.
- The schematic is supporting evidence; this is repair of an existing netlist, not image-only regeneration.
- Do not generate or rewrite the original caption.

Return exactly:
<NETLIST>
[Corrected SPICE netlist]
</NETLIST>
````
