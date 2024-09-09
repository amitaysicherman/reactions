import unittest
from preprocessing.to_edit_ops import get_edit_operations, SAVE, REMOVE, REPLACE, ADD, START_TEOKEN


class TestGetEditOperations(unittest.TestCase):
    def test_empty_source(self):
        src = START_TEOKEN
        tgt = START_TEOKEN + "abc"
        expected = [ADD]
        self.assertEqual(get_edit_operations(src, tgt), expected)

    def test_empty_target(self):
        src = START_TEOKEN + "abc"
        tgt = START_TEOKEN
        expected = [SAVE, REMOVE, REMOVE, REMOVE]
        self.assertEqual(get_edit_operations(src, tgt), expected)

    def test_equal(self):
        for l in range(1, 10):
            src = START_TEOKEN + "a" * l
            tgt = START_TEOKEN + "a" * l
            expected = [SAVE] * (l + 1)
            self.assertEqual(get_edit_operations(src, tgt), expected)
        src = "abc"
        tgt = "abc"
        expected = [SAVE, SAVE, SAVE]
        self.assertEqual(get_edit_operations(src, tgt), expected)

    def test_delete(self):
        src = START_TEOKEN + "abc"
        tgt = START_TEOKEN + "a"
        expected = [SAVE, SAVE, REMOVE, REMOVE]
        self.assertEqual(get_edit_operations(src, tgt), expected)

        src = START_TEOKEN + "abc"
        tgt = START_TEOKEN + "b"
        expected = [SAVE, REMOVE, SAVE, REMOVE]
        self.assertEqual(get_edit_operations(src, tgt), expected)

        src = START_TEOKEN + "abc"
        tgt = START_TEOKEN + "c"
        expected = [SAVE, REMOVE, REMOVE, SAVE]
        self.assertEqual(get_edit_operations(src, tgt), expected)

        src = START_TEOKEN + "ababc"
        tgt = START_TEOKEN + "bc"
        expected = [SAVE, REMOVE, REMOVE, REMOVE, SAVE, SAVE]
        self.assertEqual(get_edit_operations(src, tgt), expected)

    def test_replace(self):
        src = START_TEOKEN + "abc"
        tgt = START_TEOKEN + "adc"
        expected = [SAVE, SAVE, REPLACE, SAVE]
        self.assertEqual(get_edit_operations(src, tgt), expected)

        src = START_TEOKEN + "abc"
        tgt = START_TEOKEN + "cbc"
        expected = [SAVE, REPLACE, SAVE, SAVE]
        self.assertEqual(get_edit_operations(src, tgt), expected)

        src = START_TEOKEN + "abbc"
        tgt = START_TEOKEN + "acbc"
        expected = [SAVE, SAVE, REPLACE, SAVE, SAVE]
        self.assertEqual(get_edit_operations(src, tgt), expected)

        src = START_TEOKEN + "abbc"
        tgt = START_TEOKEN + "acc"
        expected = [SAVE, SAVE, REMOVE, REMOVE, ADD]
        self.assertEqual(get_edit_operations(src, tgt), expected)

    def test_insert(self):
        src = START_TEOKEN + "abc"
        tgt = START_TEOKEN + "abdc"
        expected = [SAVE, SAVE, ADD, SAVE]
        self.assertEqual(get_edit_operations(src, tgt), expected)

        src = START_TEOKEN + "abc"
        tgt = START_TEOKEN + "cabc"
        expected = [ADD, SAVE, SAVE, SAVE]
        self.assertEqual(get_edit_operations(src, tgt), expected)

        src = START_TEOKEN + "abc"
        tgt = START_TEOKEN + "abac"
        expected = [SAVE, SAVE, ADD, SAVE]
        self.assertEqual(get_edit_operations(src, tgt), expected)

        src = START_TEOKEN + "abc"
        tgt = START_TEOKEN + "abtttttttttttac"
        expected = [SAVE, SAVE, ADD, SAVE]
        self.assertEqual(get_edit_operations(src, tgt), expected)

        src = START_TEOKEN + "abc"
        tgt = START_TEOKEN + "abctttt"
        expected = [SAVE, SAVE, SAVE, ADD]
        self.assertEqual(get_edit_operations(src, tgt), expected)

        src = START_TEOKEN + "abc"
        tgt = START_TEOKEN + "ttttattttbtttctttt"
        expected = [ADD, ADD, ADD, ADD]
        self.assertEqual(get_edit_operations(src, tgt), expected)

    def test_combination_of_operations(self):
        src = START_TEOKEN + "abcdef"
        tgt = START_TEOKEN + "abdfgh"
        expected = [SAVE, SAVE, SAVE, REMOVE, SAVE, REMOVE, ADD]
        self.assertEqual(get_edit_operations(src, tgt), expected)

        src = START_TEOKEN + "abcdef"
        tgt = START_TEOKEN + "abcfgh"
        expected = [SAVE, SAVE, SAVE, SAVE, REMOVE, REMOVE, ADD]
        self.assertEqual(get_edit_operations(src, tgt), expected)

        src = START_TEOKEN + "abcdef"
        tgt = START_TEOKEN + "abcf"
        expected = [SAVE, SAVE, SAVE, SAVE, REMOVE, REMOVE, SAVE]
        self.assertEqual(get_edit_operations(src, tgt), expected)

        src = START_TEOKEN + "abcdef"
        tgt = START_TEOKEN + "abcfghijkl"
        expected = [SAVE, SAVE, SAVE, SAVE, REMOVE, REMOVE, ADD]
        self.assertEqual(get_edit_operations(src, tgt), expected)

        src = START_TEOKEN + "abcdef"
        tgt = START_TEOKEN + "ghijkl"
        expected = [SAVE, REPLACE, REMOVE, REMOVE, REMOVE, REMOVE, REMOVE]
        self.assertEqual(get_edit_operations(src, tgt), expected)


if __name__ == '__main__':
    unittest.main()
