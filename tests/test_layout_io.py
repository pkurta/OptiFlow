from __future__ import annotations

import unittest

from optiflow.models.layout_io import (
  INTERFACE_LAYOUT_FORMAT,
  fields_to_json,
  interface_layout_from_payload,
  interface_layout_to_payload,
  optional_weights_from_payload,
)
from optiflow.models.scoring import (
  ControlType,
  DataType,
  FieldSpec,
  build_interface_layout,
)


class InterfaceLayoutJsonTests(unittest.TestCase):
  def _sample_layout(self):
    fields = [
      FieldSpec("Возраст", DataType.UNSIGNED, 3),
      FieldSpec("Имя", DataType.TEXT, 16),
      FieldSpec("Согласие", DataType.BOOLEAN, 1),
    ]
    controls = [ControlType.SPINNER, ControlType.TEXTBOX, ControlType.CHECKBOX]
    return build_interface_layout(fields, controls, [0, 1, 1])

  def test_round_trip_preserves_controls_and_screens(self) -> None:
    layout = self._sample_layout()
    payload = interface_layout_to_payload(
      layout,
      optiflow_version="1.6",
      algorithm_key="GA",
      algorithm_label="Классический генетический алгоритм (GA)",
      weight_preset="Упор на оперативность",
      weights=(0.2, 0.7, 0.1),
      max_forms=3,
      potency=0.5,
      operativeness=0.4,
      resource_saving=0.3,
      fitness=0.42,
    )
    self.assertEqual(payload["format"], INTERFACE_LAYOUT_FORMAT)
    self.assertEqual(payload["version"], 1)
    restored = interface_layout_from_payload(payload)
    self.assertEqual(len(restored.fields), 3)
    self.assertEqual(restored.form_count, 2)
    self.assertEqual(
      [element.control for element in restored.forms[0].elements],
      [ControlType.SPINNER],
    )
    self.assertEqual(
      [element.control.name for element in restored.forms[1].elements],
      ["TEXTBOX", "CHECKBOX"],
    )
    self.assertEqual(optional_weights_from_payload(payload), (0.2, 0.7, 0.1))

  def test_rejects_problem_data_format(self) -> None:
    with self.assertRaises(ValueError) as ctx:
      interface_layout_from_payload({"format": "optiflow-problem-data", "version": 1})
    self.assertIn("optiflow-interface-layout", str(ctx.exception))

  def test_rejects_duplicate_field_index(self) -> None:
    layout = self._sample_layout()
    payload = interface_layout_to_payload(layout, optiflow_version="1.6")
    payload["forms"][1]["elements"][0]["field_index"] = 0
    with self.assertRaises(ValueError):
      interface_layout_from_payload(payload)

  def test_fields_to_json_matches_problem_data_shape(self) -> None:
    layout = self._sample_layout()
    dumped = fields_to_json(layout.fields)
    self.assertEqual(dumped[2], {"name": "Согласие", "data_type": "BOOLEAN", "size": 1})


if __name__ == "__main__":
  unittest.main()
