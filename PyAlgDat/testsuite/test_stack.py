#!/usr/bin/env py.test

"""
Test Stack class.
"""

import unittest

from py_alg_dat import stack

class TestStack(unittest.TestCase):

    """
    Test Stack class.
    """

    def setUp(self):
        self.stack1 = stack.Stack()
        self.stack1.push(10)
        self.stack1.push(20)
        self.stack1.push(30)
        self.stack1.push(40)
        self.stack1.push(50)

        self.stack2 = stack.Stack()
        self.stack2.push(10)
        self.stack2.push(20)
        self.stack2.push(30)
        self.stack2.push(40)
        self.stack2.push(50)

        self.stack3 = stack.Stack()
        self.stack3.push(100)

    def test_stack_push(self):
        """
        Test method "push".
        """
        stack1 = stack.Stack([20, 30, 40, 50])
        stack1.push(10)
        stack2 = stack.Stack([10, 20, 30, 40, 50])
        self.assertEqual(stack1, stack2)

    def test_stack_pop(self):
        """
        Test method "pop".
        """
        stack1 = stack.Stack([10, 20, 30, 40, 50])
        res = stack1.pop()
        self.assertEqual(10, res)

    def test_stack_len(self):
        """
        Test operator "len".
        """
        self.assertEqual(5, len(self.stack1))

    def test_stack_equal(self):
        """
        Test operator "equal".
        """
        self.assertEqual(self.stack1, self.stack2)

    def test_stack_not_equal(self):
        """
        Test operator "equal" - inverted.
        """
        self.assertNotEqual(self.stack2, self.stack3)

    def test_stack_is_empty(self):
        """
        Test method "is_empty".
        """
        self.assertEquals(False, self.stack1.is_empty())

    def test_stack_clear(self):
        """
        Test method "clear".
        """
        self.stack1.clear()
        self.assertEqual(0, len(self.stack1))
