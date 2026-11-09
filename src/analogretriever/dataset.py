"""SPICE netlist graph construction and the tri-modal circuit dataset (paper Sec. 3.2).

Each netlist becomes a heterogeneous graph whose nodes are devices and nets.
A device is linked to every net it touches by an edge typed with the device
terminal; devices that share a net are also linked by a ``shared_net`` edge.
"""
import functools
import hashlib
import json
import math
import os
import re
import tempfile
import warnings
from collections import ChainMap, defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

GRAPH_SCHEMA_VERSION = 1

# |R| = 20 relation types: 19 port types plus one reserved type for
# unrecognised ports.
RELATION_TYPES = (
    "unknown_port",
    "drain", "gate", "source", "bulk",            # MOSFET (bulk also used for BJT substrate)
    "collector", "base", "emitter",               # BJT
    "positive", "negative",                       # independent sources
    "anode", "cathode",                           # diode
    "terminal_1", "terminal_2",                   # R / C / L / switch
    "out_p", "out_n", "ctrl_in_p", "ctrl_in_n",   # controlled sources
    "shared_net",                                 # device-device edge on a common net
    "subckt_terminal",                            # subcircuit interface
)
RELATION_IDS = {name: index for index, name in enumerate(RELATION_TYPES)}
NUM_RELATIONS = len(RELATION_TYPES)

# Net nodes carry a generic, supply or ground net type; device nodes carry the
# device type resolved from the instance prefix and its .model declaration.
NODE_TYPES = (
    "NET", "SUPPLY_NET", "GROUND_NET",
    "RESISTOR", "CAPACITOR", "INDUCTOR", "DIODE",
    "V_SOURCE", "I_SOURCE",
    "MOSFET", "NMOS", "PMOS",
    "BJT", "NPN", "PNP",
    "JFET", "NJF", "PJF",
    "VCVS", "CCCS", "VCCS", "CCVS",
    "SWITCH", "SUBCIRCUIT",
)
NODE_TYPE_IDS = {name: index for index, name in enumerate(NODE_TYPES)}

# Continuous device parameters, in units chosen so that log(1 + x) is
# informative: MOSFET channel width and length in um, resistance in ohm and
# capacitance in pF. Net nodes and other devices use a zero vector.
CONTINUOUS_FEATURES = ("width_um", "length_um", "resistance_ohm", "capacitance_pf")
NUM_CONTINUOUS_FEATURES = len(CONTINUOUS_FEATURES)
MOSFET_TYPES = ("MOSFET", "NMOS", "PMOS")

# Instance prefix -> (default device type, number of terminals).
DEVICE_REGISTRY = {
    'R': ("RESISTOR", 2),
    'C': ("CAPACITOR", 2),
    'L': ("INDUCTOR", 2),
    'D': ("DIODE", 2),
    'V': ("V_SOURCE", 2),
    'I': ("I_SOURCE", 2),
    'M': ("MOSFET", 4),
    'Q': ("BJT", 3),
    'J': ("JFET", 3),
    'E': ("VCVS", 4),
    'F': ("CCCS", 2),
    'G': ("VCCS", 4),
    'H': ("CCVS", 2),
    'S': ("SWITCH", 4),
    'B': ("V_SOURCE", 2),
}

# Terminal names in SPICE instance order.
DEVICE_TERMINALS = {
    'R': ("terminal_1", "terminal_2"),
    'C': ("terminal_1", "terminal_2"),
    'L': ("terminal_1", "terminal_2"),
    'D': ("anode", "cathode"),
    'V': ("positive", "negative"),
    'I': ("positive", "negative"),
    'M': ("drain", "gate", "source", "bulk"),
    'Q': ("collector", "base", "emitter", "bulk"),
    'J': ("drain", "gate", "source"),
    'E': ("out_p", "out_n", "ctrl_in_p", "ctrl_in_n"),
    'F': ("out_p", "out_n"),
    'G': ("out_p", "out_n", "ctrl_in_p", "ctrl_in_n"),
    'H': ("out_p", "out_n"),
    'S': ("terminal_1", "terminal_2", "ctrl_in_p", "ctrl_in_n"),
    'B': ("positive", "negative"),
}

MODEL_POLARITIES = {'M': ("NMOS", "PMOS"), 'Q': ("NPN", "PNP"), 'J': ("NJF", "PJF")}


def parse_value(text):
    """Parse a numeric SPICE literal such as ``1uF``, ``10k`` or ``2.2MEG``.

    As in Ngspice, the scale factor is the first letter after the number and
    any following letters are an ignored unit name, so ``1uF`` is 1e-6 and
    ``1M`` is 1e-3. Symbolic parameters and expressions return 0.0.
    """
    if not text:
        return 0.0
    value = text.strip()
    if value.startswith('{') and value.endswith('}'):
        value = value[1:-1].strip()
    match = re.fullmatch(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z]*)', value)
    if match is None:
        return 0.0
    suffix = match.group(2).lower()
    if suffix.startswith('meg'):
        multiplier = 1e6
    elif suffix.startswith('mil'):
        multiplier = 25.4e-6
    else:
        multiplier = {
            't': 1e12, 'g': 1e9, 'k': 1e3, 'm': 1e-3,
            'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15, 'a': 1e-18,
        }.get(suffix[:1], 1.0)
    result = float(match.group(1)) * multiplier
    return result if math.isfinite(result) else 0.0


def netlist_records(lines):
    """Pair each logical instance line with the .model table visible to it.

    Continuation lines are joined and comments dropped. Models are collected
    before instances are parsed, so declarations may follow their use; models
    declared inside a .subckt shadow outer ones without leaking out of it.
    Subcircuits are not expanded and .include files are not loaded.
    """
    logical_lines = []
    for raw_line in lines:
        line = raw_line.split(';')[0].strip()
        if not line or line.startswith('*'):
            continue
        if line.startswith('+') and logical_lines:
            logical_lines[-1] += ' ' + line[1:].strip()
        else:
            logical_lines.append(line)

    scopes = [{}]
    records = []
    for line in logical_lines:
        directive = line.split()[0].lower()
        if directive == '.subckt':
            scopes.append({})
        elif directive == '.ends':
            if len(scopes) > 1:
                scopes.pop()
        elif directive == '.model':
            match = re.match(r'\.model\s+([^\s(),=]+)\s+([a-zA-Z]+)\b', line, flags=re.IGNORECASE)
            if match:
                scopes[-1][match.group(1).upper()] = match.group(2).upper()
        else:
            records.append((line, ChainMap(*reversed(scopes))))
    return records


def bjt_terminal_count(parts, models):
    """Distinguish ``Q c b e model [area]`` from ``Q c b e substrate model``."""
    bjt_types = ("NPN", "PNP")
    for model_index in (4, 5):
        if len(parts) > model_index and models.get(parts[model_index].upper()) in bjt_types:
            return model_index - 1
    for model_index in (4, 5):
        if len(parts) > model_index and parts[model_index].upper() in bjt_types:
            return model_index - 1
    return 3


def is_net_token(token):
    """Net names exclude ``key=value`` parameters, ``{expr}`` values and DC/AC keywords."""
    return bool(token) and '=' not in token and not token.startswith('{') \
        and token.upper() not in ('DC', 'AC')


def net_type(net_name):
    if "VDD" in net_name or "VCC" in net_name:
        return "SUPPLY_NET"
    if "VSS" in net_name or "GND" in net_name or net_name == "0":
        return "GROUND_NET"
    return "NET"


def device_features(base_type, parameters):
    """Continuous feature vector in the units of ``CONTINUOUS_FEATURES``."""
    width = length = resistance = capacitance = 0.0
    for token in parameters:
        token = token.strip('{}')
        upper = token.upper()
        if base_type in MOSFET_TYPES and upper.startswith("W="):
            width = parse_value(token[2:]) * 1e6
        elif base_type in MOSFET_TYPES and upper.startswith("L="):
            length = parse_value(token[2:]) * 1e6
        elif base_type == "RESISTOR" and '=' not in token:
            value = parse_value(token)
            if value != 0.0:
                resistance = value
        elif base_type == "CAPACITOR" and '=' not in token:
            value = parse_value(token)
            if value != 0.0:
                capacitance = value * 1e12
    return [width, length, resistance, capacitance]


def parse_netlist(text, source="<netlist>"):
    """Convert SPICE text into ``(x_type, x_continuous, edge_index, edge_type)`` tensors.

    Current-controlled sources (F/H, and ``I(Vx)`` terms of B sources) are
    linked through ``ctrl_in_p``/``ctrl_in_n`` edges to the two nets of the
    sensing voltage source; ``V(a, b)`` terms of B sources are linked to nets
    ``a`` and ``b`` in the same way. Device and net names are case-insensitive
    and local to their .subckt scope, except ground.
    """
    node_index = {}
    node_types = []
    node_features = []
    edges_source, edges_target, edges_type = [], [], []
    net_devices = defaultdict(list)
    device_nets = {}
    current_controls = []
    voltage_controls = []
    unsupported = set()

    def add_node(key, type_name, features=None):
        if key not in node_index:
            node_index[key] = len(node_index)
            node_types.append(NODE_TYPE_IDS[type_name])
            node_features.append(features or [0.0] * NUM_CONTINUOUS_FEATURES)
        return node_index[key]

    def add_net(scope, net_name):
        key = ('net', None if net_name == '0' else scope, net_name)
        return key, add_node(key, net_type(net_name))

    def connect(device, net_key, net, relation):
        edges_source.extend([device, net])
        edges_target.extend([net, device])
        edges_type.extend([RELATION_IDS[relation]] * 2)
        if device not in net_devices[net_key]:
            net_devices[net_key].append(device)

    for line, models in netlist_records(text.splitlines()):
        if line.startswith('.') or line.lower() == 'spice':
            continue
        parts = line.split()
        if len(parts) < 3:
            continue

        device_name = parts[0].upper()
        prefix = device_name[0]
        scope = id(models.maps[0])

        if prefix in DEVICE_REGISTRY:
            base_type, num_terminals = DEVICE_REGISTRY[prefix]
            terminals = DEVICE_TERMINALS[prefix]
        elif prefix == 'X':
            base_type, num_terminals = "SUBCIRCUIT", len(parts) - 2
            terminals = ("subckt_terminal",) * num_terminals
        else:
            unsupported.add(prefix)
            continue
        if prefix == 'Q':
            num_terminals = bjt_terminal_count(parts, models)

        nets = []
        for token in parts[1:]:
            if len(nets) == num_terminals or not is_net_token(token):
                break
            nets.append(token.upper())
        parameters = parts[1 + len(nets):]

        if prefix in ('F', 'H'):
            if len(nets) != 2 or len(parameters) < 2 or not parameters[0].upper().startswith('V'):
                raise ValueError(f"Unsupported F/H control syntax in {source}: {line}")
            current_controls.append((scope, device_name, parameters[0].upper()))
        elif prefix == 'B':
            match = re.match(r'([IV])\s*=\s*(.+)', ' '.join(parameters), flags=re.IGNORECASE)
            if match is None:
                raise ValueError(f"Unsupported behavioral-source syntax in {source}: {line}")
            base_type = "I_SOURCE" if match.group(1).upper() == 'I' else "V_SOURCE"
            expression = match.group(2)
            voltages = re.findall(r'\bV\s*\(\s*([^(),\s]+)\s*(?:,\s*([^(),\s]+)\s*)?\)',
                                  expression, flags=re.IGNORECASE)
            currents = re.findall(r'\bI\s*\(\s*([^(),\s]+)\s*\)', expression, flags=re.IGNORECASE)
            if len(voltages) + len(currents) != len(re.findall(r'\b[VI]\s*\(', expression,
                                                               flags=re.IGNORECASE)):
                warnings.warn(f"{source}: some behavioral-source dependencies are unsupported: {line}",
                              RuntimeWarning, stacklevel=2)
            for positive, negative in voltages:
                voltage_controls.append((scope, device_name, positive.upper(), (negative or '0').upper()))
            for sensing_source in currents:
                current_controls.append((scope, device_name, sensing_source.upper()))
        elif prefix in MODEL_POLARITIES and parameters:
            model_name = parameters[0].upper().strip('{}')
            model_type = models.get(model_name, model_name)
            if model_type in MODEL_POLARITIES[prefix]:
                base_type = model_type

        device_key = ('device', scope, device_name)
        if device_key in node_index:
            raise ValueError(f"Duplicate device name {device_name!r} in {source}")
        device = add_node(device_key, base_type, device_features(base_type, parameters))
        device_nets[(scope, device_name)] = []

        for position, net_name in enumerate(nets):
            net_key, net = add_net(scope, net_name)
            relation = terminals[position] if position < len(terminals) else "unknown_port"
            connect(device, net_key, net, relation)
            device_nets[(scope, device_name)].append((net_key, net))

    for scope, device_name, sensing_source in dict.fromkeys(current_controls):
        sensing_nets = device_nets.get((scope, sensing_source))
        if sensing_nets is None or len(sensing_nets) != 2:
            raise ValueError(f"Unresolved control source {sensing_source!r} for {device_name} in {source}")
        device = node_index[('device', scope, device_name)]
        for (net_key, net), relation in zip(sensing_nets, ("ctrl_in_p", "ctrl_in_n")):
            connect(device, net_key, net, relation)

    for scope, device_name, positive, negative in dict.fromkeys(voltage_controls):
        device = node_index[('device', scope, device_name)]
        for net_name, relation in ((positive, "ctrl_in_p"), (negative, "ctrl_in_n")):
            net_key, net = add_net(scope, net_name)
            connect(device, net_key, net, relation)

    shared_net = RELATION_IDS["shared_net"]
    for devices in net_devices.values():
        for i in range(len(devices)):
            for j in range(i + 1, len(devices)):
                edges_source.extend([devices[i], devices[j]])
                edges_target.extend([devices[j], devices[i]])
                edges_type.extend([shared_net, shared_net])

    if unsupported:
        warnings.warn(f"{source}: unsupported device prefixes {sorted(unsupported)} were omitted",
                      RuntimeWarning, stacklevel=2)
    if not node_index:
        raise ValueError(f"No supported circuit devices found in {source}")

    return (torch.tensor(node_types, dtype=torch.long),
            torch.tensor(node_features, dtype=torch.float),
            torch.tensor([edges_source, edges_target], dtype=torch.long),
            torch.tensor(edges_type, dtype=torch.long))


def callable_identity(function):
    if isinstance(function, functools.partial):
        return [callable_identity(function.func), repr(function.args), repr(sorted(function.keywords.items()))]
    return f"{getattr(function, '__module__', '')}.{getattr(function, '__qualname__', type(function).__qualname__)}"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class TriModalCircuitDataset(InMemoryDataset):
    """Schematic image, description tokens and netlist graph for every CSV row.

    The CSV needs ``id``, ``image``, ``spice`` and ``caption`` columns, with
    paths relative to ``root``. Graphs and caption tokens are cached under
    ``root/processed``; the cache is rebuilt whenever an input file, the CSV
    or this module changes. Images are loaded and transformed on every access.
    """

    def __init__(self, root, csv_file, tokenizer, image_preprocess, transform=None):
        self.csv_file = str(csv_file)
        self.df = pd.read_csv(self.csv_file)
        if self.df.empty or not {'id', 'image', 'spice', 'caption'}.issubset(self.df.columns):
            raise ValueError("A nonempty CSV with id, image, spice and caption columns is required")
        self.tokenizer = tokenizer
        self.image_preprocess = image_preprocess
        self._input_root = os.path.abspath(root)

        cache_path = Path(root) / 'processed' / self.processed_file_names[0]
        self._metadata_path = cache_path.with_suffix('.metadata.json')
        self._fingerprint = self._input_fingerprint()
        try:
            recorded = json.loads(self._metadata_path.read_text())['fingerprint']
        except (OSError, ValueError, KeyError, TypeError):
            recorded = None
        super().__init__(root, transform,
                         force_reload=cache_path.exists() and recorded != self._fingerprint)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.csv_file)]

    @property
    def processed_file_names(self):
        return [f'{Path(self.csv_file).stem}_graphs_v{GRAPH_SCHEMA_VERSION}.pt']

    def _row_paths(self, row):
        return (os.path.join(self._input_root, str(row['image'])),
                os.path.join(self._input_root, str(row['spice'])))

    def _input_fingerprint(self):
        files = []
        for _, row in self.df.iterrows():
            if pd.isna(row['caption']) or not str(row['caption']).strip():
                raise ValueError(f"Blank caption for circuit {row['id']}")
            files.append([file_sha256(path) for path in self._row_paths(row)])
        payload = {'schema': GRAPH_SCHEMA_VERSION, 'parser': file_sha256(__file__),
                   'csv': file_sha256(self.csv_file), 'files': files,
                   'tokenizer': callable_identity(self.tokenizer)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def process(self):
        data_list = []
        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Building netlist graphs"):
            _, netlist_path = self._row_paths(row)
            try:
                text = Path(netlist_path).read_text(encoding='utf-8')
                x_type, x_continuous, edge_index, edge_type = parse_netlist(text, netlist_path)
            except Exception as error:
                raise ValueError(f"Failed to preprocess CSV row {index} ({netlist_path}): {error}") from error
            data_list.append(Data(
                x_type=x_type, x_continuous=x_continuous,
                edge_index=edge_index, edge_attr=edge_type,
                text_tokens=self.tokenizer([str(row['caption'])], truncate=True),
                image_rel_path=str(row['image']), circuit_id=str(row['id']),
                num_nodes=x_type.size(0),
            ))

        # Write atomically so that an interrupted build is never reused.
        descriptor, temporary = tempfile.mkstemp(dir=self.processed_dir, suffix='.pt')
        os.close(descriptor)
        try:
            self.save(data_list, temporary)
            os.replace(temporary, self.processed_paths[0])
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        metadata = {'fingerprint': self._fingerprint, 'rows': len(data_list),
                    'graph_schema_version': GRAPH_SCHEMA_VERSION}
        self._metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    def get(self, idx):
        data = super().get(idx)
        with Image.open(os.path.join(self._input_root, data.image_rel_path)) as image:
            data.image_tensor = self.image_preprocess(image.convert("RGB")).unsqueeze(0)
        return data
