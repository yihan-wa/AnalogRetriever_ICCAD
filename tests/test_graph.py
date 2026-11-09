"""SPICE parsing and netlist graph construction (paper Sec. 3.2)."""
import math
from pathlib import Path
import sys
import unittest
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    from analogretriever.dataset import (NODE_TYPES, NUM_RELATIONS, RELATION_IDS, RELATION_TYPES,
                                         parse_netlist, parse_value)
except ImportError:
    parse_netlist = None


def relation(name):
    return RELATION_IDS[name]


@unittest.skipIf(parse_netlist is None, "Dataset dependencies are required")
class GraphChecks(unittest.TestCase):
    def parse(self, text):
        return parse_netlist("* fixture\n" + text + "\n.end\n")

    def device_types(self, graph):
        return [NODE_TYPES[i] for i in graph[0].tolist() if not NODE_TYPES[i].endswith("NET")]

    def edges(self, graph, name):
        return sorted(graph[2][:, graph[3] == relation(name)].t().tolist())

    def test_twenty_relation_types_with_reserved_unknown_port(self):
        self.assertEqual(NUM_RELATIONS, 20)
        self.assertEqual(RELATION_TYPES[0], "unknown_port")
        self.assertEqual(len(set(RELATION_TYPES)), 20)

    def test_numeric_scale_and_unit_suffixes(self):
        cases = {
            "1uF": 1e-6, "10mH": 0.01, "3000PF": 3e-9, "10kOhm": 1e4, "1M": 1e-3,
            "1MEG": 1e6, "2.2megOhm": 2.2e6, "1mil": 25.4e-6, ".5u": 5e-7,
            "-2.5mA": -2.5e-3, "+3V": 3.0, "1e-3": 1e-3, "2.5e3Ohm": 2500.0,
            "1e3uF": 1e-3, "{2.2u}": 2.2e-6, "0": 0.0,
        }
        for text, expected in cases.items():
            with self.subTest(value=text):
                self.assertTrue(math.isclose(parse_value(text), expected, rel_tol=1e-12, abs_tol=0.0))

    def test_symbolic_values_are_zero(self):
        for text in ("", "R1", "{R1}", "{C1}", "NPNPWR", "{1+2}"):
            with self.subTest(value=text):
                self.assertEqual(parse_value(text), 0.0)

    def test_continuous_features_use_paper_parameters_and_units(self):
        graph = self.parse("C1 1 0 1uF\nL1 1 2 10mH\nR1 2 0 10kOhm\nM1 d g s b NMOS W=2u L=.18u\n"
                           "S1 a b c d SW W=5")
        features = dict(zip(self.device_types(graph),
                            graph[1][[i for i, t in enumerate(graph[0].tolist())
                                      if not NODE_TYPES[t].endswith("NET")]].tolist()))
        expected = {"CAPACITOR": [0, 0, 0, 1e6], "INDUCTOR": [0, 0, 0, 0], "RESISTOR": [0, 0, 1e4, 0],
                    "NMOS": [2, 0.18, 0, 0], "SWITCH": [0, 0, 0, 0]}
        for device, values in expected.items():
            with self.subTest(device=device):
                for actual, value in zip(features[device], values):
                    self.assertTrue(math.isclose(actual, value, rel_tol=1e-6, abs_tol=1e-9), features[device])

    def test_net_nodes_have_generic_supply_or_ground_type(self):
        graph = self.parse("R1 VDD out 1k\nR2 out 0 1k")
        self.assertEqual([NODE_TYPES[i] for i in graph[0].tolist()],
                         ["RESISTOR", "SUPPLY_NET", "NET", "RESISTOR", "GROUND_NET"])
        self.assertTrue(graph[1][[1, 2, 4]].eq(0).all())

    def test_model_declarations_set_device_polarity(self):
        graph = self.parse("Q1 c1 b1 e1 NPNPWR\nQ2 c2 b2 e2 PNPMOD\nQ3 c3 b3 e3 NPN\n"
                           "M1 d g s body NMOS_NAME W=1u L=.18u\n"
                           ".model NPNPWR NPN (BF=100)\n.model PNPMOD NPN (BF=100)\n"
                           ".model NPN PNP (BF=80)\n.model NMOS_NAME PMOS (LEVEL=1)")
        self.assertEqual(self.device_types(graph), ["NPN", "NPN", "PNP", "PMOS"])

    def test_models_resolve_case_insensitively_after_use_and_on_continuation_lines(self):
        graph = self.parse("Q1 c1 b1 e1 q2n3904\nQ2 c2 b2 e2 qpart\nM1 d g s b vn10lmmod\n"
                           ".MoDeL Q2N3904 NPN(BF=100)\n.model qpart\n+ PNP (BF=100)\n.MODEL VN10LMMOD NMOS")
        self.assertEqual(self.device_types(graph), ["NPN", "PNP", "NMOS"])

    def test_undeclared_models_keep_generic_type(self):
        graph = self.parse("Q1 c b e NPNPWR\nM1 d g s body PMOS_PART W=1u L=.18u")
        self.assertEqual(self.device_types(graph), ["BJT", "MOSFET"])

    def test_subcircuit_models_do_not_leak(self):
        graph = self.parse("Qbefore c1 b1 e1 m\n.subckt cell a b c\nQinside a b c m\n.model m PNP\n"
                           ".ends cell\nQafter c2 b2 e2 m\n.model m NPN")
        self.assertEqual(self.device_types(graph), ["NPN", "PNP", "NPN"])

    def test_bjt_terminal_count(self):
        three = self.parse("Q1 c b e qmod 2 OFF\n.model qmod NPN")
        self.assertEqual(three[0].numel(), 4)
        self.assertEqual(self.edges(three, "bulk"), [])
        four = self.parse("Q1 c b e sub qmod AREA=2\n.model qmod PNP")
        self.assertEqual(four[0].numel(), 5)
        self.assertEqual(self.edges(four, "bulk"), [[0, 4], [4, 0]])

    def test_current_controlled_source_connects_to_sensing_source_nets(self):
        for prefix in ("F", "H"):
            with self.subTest(prefix=prefix):
                graph = self.parse(f"{prefix}1 out 0 vsense 10\nVSENSE input 0 0")
                # Nodes: F1, OUT, 0, VSENSE, INPUT.
                self.assertEqual(graph[0].numel(), 5)
                self.assertEqual(self.edges(graph, "ctrl_in_p"), [[0, 4], [4, 0]])
                self.assertEqual(self.edges(graph, "ctrl_in_n"), [[0, 2], [2, 0]])

    def test_control_reference_changes_graph_and_missing_reference_fails(self):
        a = self.parse("V1 1 0 0\nV2 2 0 0\nF1 3 0 V1 10")
        b = self.parse("V1 1 0 0\nV2 2 0 0\nF1 3 0 V2 10")
        self.assertFalse(a[2].equal(b[2]))
        with self.assertRaisesRegex(ValueError, "Unresolved control source"):
            self.parse("F1 out 0 Vmissing 10")

    def test_behavioral_source_dependencies(self):
        graph = self.parse("B1 out 0 V = {V(a,b) + v(a,b) + V(c)}")
        self.assertEqual(self.device_types(graph), ["V_SOURCE"])
        self.assertEqual(len(self.edges(graph, "ctrl_in_p")), 4)
        self.assertEqual(len(self.edges(graph, "ctrl_in_n")), 4)
        current = self.parse("VS x 0 0\nB2 out 0 I = {2*I(VS)}")
        self.assertEqual(self.device_types(current), ["V_SOURCE", "I_SOURCE"])
        self.assertEqual(self.edges(current, "ctrl_in_p"), [[1, 3], [3, 1]])

    def test_switch_and_jfet_ports(self):
        switch = self.parse("S1 a b cp cn SWMOD\n.model SWMOD SW(RON=1)")
        self.assertEqual(sorted(set(switch[3].tolist())),
                         sorted(relation(r) for r in ("terminal_1", "terminal_2", "ctrl_in_p", "ctrl_in_n")))
        for model_type in ("NJF", "PJF"):
            graph = self.parse(f"J1 d g s part\n.model part {model_type}")
            self.assertEqual(self.device_types(graph), [model_type])
            self.assertEqual(sorted(set(graph[3].tolist())), [relation("drain"), relation("gate"), relation("source")])

    def test_mutual_inductance_lines_are_omitted(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            coupled = self.parse("K1 L1 L2 .98\nL1 a 0 1m\nL2 b 0 2m")
        self.assertTrue(any("['K']" in str(item.message) for item in caught))
        plain = self.parse("L1 a 0 1m\nL2 b 0 2m")
        for left, right in zip(coupled, plain):
            self.assertTrue(left.equal(right))

    def test_names_are_case_insensitive_and_devices_separate_from_nets(self):
        graph = self.parse("V1 V1 0 1")
        self.assertEqual(graph[0].numel(), 3)
        shared = self.parse("R1 Out 0 1k\nR2 OUT 2 2k")
        self.assertEqual(shared[0].numel(), 5)
        self.assertEqual(len(self.edges(shared, "shared_net")), 2)

    def test_duplicate_devices_fail_and_tied_terminals_add_no_self_edges(self):
        with self.assertRaisesRegex(ValueError, "Duplicate device"):
            self.parse("R1 1 0 1k\nr1 2 0 2k")
        graph = self.parse("M1 d g 0 0 NMOS\nR1 d 0 1k")
        shared = graph[2][:, graph[3] == relation("shared_net")]
        self.assertTrue((shared[0] != shared[1]).all())
        self.assertEqual(shared.shape[1], 4)

    def test_subcircuit_instances_use_subckt_terminal(self):
        graph = self.parse("X1 in out vdd 0 opamp")
        self.assertEqual(self.device_types(graph), ["SUBCIRCUIT"])
        self.assertEqual(set(graph[3].tolist()), {relation("subckt_terminal")})

    def test_empty_netlist_fails(self):
        with self.assertRaisesRegex(ValueError, "No supported circuit devices"):
            self.parse(".title no devices")


if __name__ == "__main__":
    unittest.main()
