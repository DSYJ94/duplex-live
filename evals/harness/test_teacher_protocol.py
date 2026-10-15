import unittest
from evals.teacher_protocol import CONTROL_PROTOCOL, CONTROL_INSTRUCTIONS, validate_protocol


class TeacherProtocolTests(unittest.TestCase):
    def test_explicit_control_protocol_is_scoped_to_control_teachers(self):
        for kind in ("clarify", "cancel"):
            self.assertEqual(validate_protocol(CONTROL_PROTOCOL, "teacher", kind), CONTROL_INSTRUCTIONS)
        for name, purpose, kind in ((CONTROL_PROTOCOL, "comparison", "cancel"),
                (CONTROL_PROTOCOL, "teacher", "private"), (None, "teacher", "clarify")):
            with self.assertRaises(ValueError):
                validate_protocol(name, purpose, kind)


if __name__ == "__main__":
    unittest.main()
