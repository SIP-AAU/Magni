"""
..
    Copyright (c) 2016, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Module providing unittests for `magni.utils.validation`.

"""

from __future__ import division
import unittest

import numpy as np

import magni
from magni.utils.validation import *


class AbstractTest(object):
    def _test_invalid_argument(self, args, kwargs, value):
        for kwarg in kwargs:
            def example(var):
                @decorate_validation
                def validate_input():
                    self._validate('var', *args, **kwarg)

                validate_input()

            behaves_correctly = False

            try:
                example(value)
            except Exception as e:
                if e.args[0][:24] == 'Invalid validation call:':
                    behaves_correctly = True
                else:
                    print('\nException caught:\n    {}'.format(e.args[0]))

            if not behaves_correctly:
                kwarg = list(kwarg.keys()) + list(kwarg.values())
                self.fail('{!r} should not accept kwarg, {}={!r}.'
                          .format(self._validate.__name__, *kwarg))

    def _test_invalid_value(self, args, kwarg, values, exc):
        def example(var):
            @decorate_validation
            def validate_input():
                self._validate('var', *args, **kwarg)

            validate_input()

        for value in values:
            behaves_correctly = False

            try:
                example(value)
            except exc as e:
                if e.args[0][:24] == 'Invalid validation call:':
                    raise
                else:
                    behaves_correctly = True

            if not behaves_correctly:
                self.fail('{!r} should fail here.'
                          .format(self._validate.__name__))

    def _test_valid_argument(self, args, kwargs, value):
        for kwarg in kwargs:
            def example(var):
                @decorate_validation
                def validate_input():
                    self._validate('var', *args, **kwarg)

                validate_input()

            behaves_correctly = True

            try:
                example(value)
            except Exception as e:
                if e.args[0][:24] == 'Invalid validation call:':
                    behaves_correctly = False
                else:
                    raise

            if not behaves_correctly:
                kwarg = list(kwarg.keys()) + list(kwarg.values())
                self.fail('{!r} should accept kwarg {}={!r}.'
                          .format(self._validate.__name__, *kwarg))

    def _test_valid_value(self, args, kwarg, values):
        def example(var):
            @decorate_validation
            def validate_input():
                self._validate('var', *args, **kwarg)

            validate_input()

        for value in values:
            try:
                example(value)
            except Exception as e:
                print('\nException caught:\n    {}'.format(e.args[0]))
                self.fail('{!r} should not fail here.'
                          .format(self._validate.__name__))


class TestCommon(unittest.TestCase):
    """
    Test of common validation function functionality.

    Implemented tests:

    * test_ignore_none : ensure that all validation functions can ignore none
      values if flagged to do so.
    * test_variable_passing : ensure that function arguments can be passed
      directly to validation functions to circumvent automatic function
      argument retrieval.
    * test_variable_retrieval : ensure that all validation functions can
      retrieve function arguments automatically.

    """

    _validators = (validate_generic, validate_numeric, validate_levels)

    def test_ignore_none(self):
        for validate in TestCommon._validators:
            if validate is validate_generic:
                arg = 'string'
            elif validate is validate_levels:
                continue
            elif validate is validate_numeric:
                arg = 'integer',
            else:
                raise ValueError('Unknown validator: {!r}'
                                 .format(validate.__name__))

            self._test_ignore_none(validate, arg)

    def test_variable_passing(self):
        for validate in TestCommon._validators:
            if validate is validate_generic:
                arg = 'string'
                var = ''
            elif validate is validate_levels:
                continue
            elif validate is validate_numeric:
                arg = 'integer',
                var = 0
            else:
                raise ValueError('Unknown validator: {!r}'
                                 .format(validate.__name__))

            self._test_variable_passing(validate, arg, var)

    def test_variable_retrieval(self):
        for validate in TestCommon._validators:
            if validate is validate_generic:
                arg = 'string'
                var = ''
            elif validate is validate_levels:
                arg = (validate_generic(None, 'string'),)
                var = ''
            elif validate is validate_numeric:
                arg = 'integer',
                var = 0
            else:
                raise ValueError('Unknown validator: {!r}'
                                 .format(validate.__name__))

            self._test_variable_retrieval(validate, arg, var)

    def _test_ignore_none(self, validate, arg):
        def example(var):
            @decorate_validation
            def validate_input():
                validate('var', arg)

            validate_input()

        def example_ignore(var):
            @decorate_validation
            def validate_input():
                validate('var', arg, ignore_none=True)

            validate_input()

        try:
            example(None)
        except ValueError as e:
            if e.args[0][:24] == 'Invalid validation call:':
                raise
        else:
            self.fail('{!r} should not blindly ignore {!r}.'
                      .format(validate.__name__, None))

        try:
            example_ignore(None)
        except ValueError as e:
            print('\nException caught:\n    {}'.format(e.args[0]))
            self.fail('{!r} should be able to ignore {!r}.'
                      .format(validate.__name__, None))

    def _test_variable_passing(self, validate, arg, var):
        def example(var):
            @decorate_validation
            def validate_input():
                validate('some_unlikely_name', arg, var=var)

            validate_input()

        try:
            example(var)
        except NameError as e:
            print('\nException caught:\n    {}'.format(e.args[0]))
            self.fail('{!r} should use passed function arguments.'
                      .format(decorate_validation.__name__))

    def _test_variable_retrieval(self, validate, arg, var):
        def example(var, var_in_sequence, var_in_mapping):
            @decorate_validation
            def validate_input():
                validate('var', arg)
                validate(('var_in_sequence', 0), arg)
                validate(('var_in_mapping', 'key'), arg)

            validate_input()

        def example_nonsense_name(var):
            @decorate_validation
            def validate_input():
                validate('no such var', arg)

            validate_input()

        def example_nonsense_index(var):
            @decorate_validation
            def validate_input():
                validate(('var', 'no such index'), arg)

            validate_input()

        try:
            example(var, [var], {'key': var})
        except (NameError, LookupError) as e:
            print('\nException caught:\n    {}'.format(e.args[0]))
            self.fail('{!r} should be able to locate the argument.'
                      .format(validate.__name__))

        try:
            example_nonsense_name(var)
        except NameError:
            pass
        else:
            self.fail('{!r} should not be able to locate this argument.'
                      .format(validate.__name__))

        try:
            example_nonsense_index(var)
        except LookupError:
            pass
        else:
            self.fail('{!r} should not be able to locate this argument.'
                      .format(validate.__name__))


class TestGeneric(unittest.TestCase, AbstractTest):
    """
    Test of validate_generic.

    Implemented tests:

    * test_has_keys : ensure that the function only accepts certain types for
      the has_keys argument, and that key checking works as intended.
    * test_keys_in : ensure that the function only accepts certain types for
      the keys_in argument, and that key checking works as intended.
    * test_len : ensure that the function only accepts certain types for the
      len argument, and that length checking works as intended.
    * test_superclass : ensure that the function only accepts certain types for
      the superclass argument, and that superclass checking works as intended.
    * test_type : ensure that the function only accepts certain types for the
      type_ argument, and that type checking works as intended.
    * test_value_in : ensure that the function only accepts certain types for
      the value_in argument, and that value checking works as intended.

    """

    def setUp(self):
        self._validate = validate_generic

    def test_has_keys(self):
        args = ('mapping',)

        value = {'key': 0}
        kwargs = ({'has_keys': ('key',)},)
        self._test_valid_argument(args, kwargs, value)

        kwargs = ({'has_keys': ''}, {'has_keys': 0})
        self._test_invalid_argument(args, kwargs, value)

        kwarg = {'has_keys': ('key',)}
        values = ({'key': 0, 'another_key': 0},)
        self._test_valid_value(args, kwarg, values)

        values = ({'unknown_key': 0, 'another_key': 0},)
        self._test_invalid_value(args, kwarg, values, KeyError)

    def test_keys_in(self):
        args = ('mapping',)

        value = {'key': 0}
        kwargs = ({'keys_in': ('key',)},)
        self._test_valid_argument(args, kwargs, value)

        kwargs = ({'keys_in': ''}, {'keys_in': 0})
        self._test_invalid_argument(args, kwargs, value)

        kwarg = {'keys_in': ('key', 'another_key')}
        values = ({'key': 0}, {'another_key': 0},)
        self._test_valid_value(args, kwarg, values)

        values = ({'unknown_key': 0, 'another_key': 0},)
        self._test_invalid_value(args, kwarg, values, KeyError)

    def test_len(self):
        args = ('explicit collection',)

        value = [0]
        kwargs = ({'len_': 1},)
        self._test_valid_argument(args, kwargs, value)

        kwargs = ({'len_': ''}, {'len_': ()})
        self._test_invalid_argument(args, kwargs, value)

        kwarg = {'len_': 3}
        values = ([0, 1, 2], (0, 1, 2))
        self._test_valid_value(args, kwarg, values)

        values = ([], (0, 1, 2, 3, 4))
        self._test_invalid_value(args, kwarg, values, ValueError)

    def test_superclass(self):
        args = ('class',)

        value = str
        kwargs = ({'superclass': object},)
        self._test_valid_argument(args, kwargs, value)

        kwargs = ({'superclass': object()}, {'superclass': (str, int)})
        self._test_invalid_argument(args, kwargs, value)

        kwarg = {'superclass': magni.utils.validation.types.MatrixBase}
        values = (magni.utils.matrices.Matrix,)
        self._test_valid_value(args, kwarg, values)

        values = (object, str)
        self._test_invalid_value(args, kwarg, values, TypeError)

    def test_type(self):
        args = ()

        value = ''
        kwargs = ({'type_': 'string'}, {'type_': str}, {'type_': (str, int)})
        self._test_valid_argument(args, kwargs, value)

        kwargs = ({'type_': 'unknown type'}, {'type_': 0})
        self._test_invalid_argument(args, kwargs, value)

        kwarg = {'type_': 'explicit collection'}
        values = ([], ())
        self._test_valid_value(args, kwarg, values)

        values = ('', 0, {})
        self._test_invalid_value(args, kwarg, values, TypeError)

    def test_value_in(self):
        args = ('string',)

        value = 'yes'
        kwargs = ({'value_in': ('yes',)},)
        self._test_valid_argument(args, kwargs, value)

        kwargs = ({'value_in': 0}, {'value_in': {}})
        self._test_invalid_argument(args, kwargs, value)

        kwarg = {'value_in': ('yes', 'no')}
        values = ('yes', 'no')
        self._test_valid_value(args, kwarg, values)

        values = ('maybe',)
        self._test_invalid_value(args, kwarg, values, ValueError)


class TestLevels(unittest.TestCase, AbstractTest):
    """
    Test of validate_levels.

    Implemented tests:

    * test_levels : ensure that the function only accepts valid level
      specifications for the levels argument, and that level checking works as
      intended.

    """

    def setUp(self):
        self._validate = validate_levels

    def test_levels(self):
        args = ()

        value = ('some', 'values')
        kwargs = (
            {'levels': (validate_generic(None, tuple),)},
            {'levels': (
                validate_generic(None, 'explicit collection'),
                validate_generic(None, 'string'))})
        self._test_valid_argument(args, kwargs, value)

        kwargs = (
            {'levels': ''},
            {'levels': ()},
            {'levels': ('',)},
            {'levels': ((),)},
            {'levels': (('levels',),)})
        self._test_invalid_argument(args, kwargs, value)

        kwarg = {'levels': (
            validate_generic(None, 'explicit collection'),
            validate_generic(None, 'string'))}
        values = (('some', 'values'), [], ['', '', ''])
        self._test_valid_value(args, kwarg, values)

        values = ({'key': 'value'}, (0, 1, 2), [{}, (), []])
        self._test_invalid_value(args, kwarg, values, TypeError)


class TestNumeric(unittest.TestCase, AbstractTest):
    """
    Test of validate_numeric.

    Implemented tests:

    * test_precision : ensure that the function only accepts known bit-lengths
      for the precision argument, and that precision checking works as
      intended.
    * test_range : ensure that the function only accepts valid string
      specifications for the range_ argument, and that range checking works as
      intended.
    * test_shape : ensure that the function only accepts valid types for the
      shape argument, and that shape chekcing works as intended.
    * test_type : ensure that the function only accepts certain types for the
      type_ argument, and that type checking works as intended.

    """

    def setUp(self):
        self._validate = validate_numeric

    def test_precision(self):
        args = ('integer',)

        value = np.int64()
        kwargs = ({'precision': 64}, {'precision': (32, 64)})
        self._test_valid_argument(args, kwargs, value)

        kwargs = ({'precision': ''}, {'precision': 42})
        self._test_invalid_argument(args, kwargs, value)

        kwarg = {'precision': (32, 64)}
        values = (np.int32(), np.int64())
        self._test_valid_value(args, kwarg, values)

        values = (np.int8(), np.int16())
        self._test_invalid_value(args, kwarg, values, TypeError)

    def test_range(self):
        args = ('integer',)

        value = 1
        kwargs = ({'range_': '[0;inf]'}, {'range_': '(0;inf)'},
                  {'range_': ']0;inf['}, {'range_': ('[0;inf]', '[0;0]')})
        self._test_valid_argument(args, kwargs, value)

        kwargs = ({'range_': ')0;inf('}, {'range_': '[0,inf]'},
                  {'range_': (0, 10)}, {'range_': ('[0;1]', '[0;1]', ['0;1'])})
        self._test_invalid_argument(args, kwargs, value)

        kwarg = {'range_': ']0;inf]'}
        values = (1, np.iinfo(np.int64).max)
        self._test_valid_value(args, kwarg, values)

        values = (0, np.iinfo(np.int64).min)
        self._test_invalid_value(args, kwarg, values, ValueError)

    def test_shape(self):
        args = ('floating',)

        value = np.zeros((5, 5))
        kwargs = ({'shape': (5, 5)}, {'shape': (5, -1)})
        self._test_valid_argument(args, kwargs, value)

        kwargs = ({'shape': ''}, {'shape': 0}, {'shape': ('', '')})
        self._test_invalid_argument(args, kwargs, value)

        kwarg = {'shape': (5, -1)}
        values = (np.zeros((5, 5)), np.zeros((5, 10)))
        self._test_valid_value(args, kwarg, values)

        values = (0., np.zeros((5, 5, 5)), np.zeros((10, 5)))
        self._test_invalid_value(args, kwarg, values, ValueError)

    def test_type(self):
        args = ()

        value = 0
        kwargs = ({'type_': 'integer'}, {'type_': ('integer', 'floating')})
        self._test_valid_argument(args, kwargs, value)

        kwargs = ({'type_': 'unknown type'}, {'type_': 0}, {'type_': int})
        self._test_invalid_argument(args, kwargs, value)

        kwarg = {'type_': ('integer', 'floating')}
        values = (0, 1., np.int8())
        self._test_valid_value(args, kwarg, values)

        values = ('', 1j, np.bool8())
        self._test_invalid_value(args, kwarg, values, TypeError)


class TestUtils(unittest.TestCase):
    """
    Test of validation utility functions.

    Implemented tests:

    * test_disable_validation : ensure that all validation functions can be
      disabled.
    * test_require_decoration : ensure that all validation functions do not run
      if they are not decorated by decorate_validation.
    * test_validate_once : ensure that all validation functions run only once
      when validate_once is enabled.

    """

    _validators = (validate_generic, validate_numeric, validate_levels)

    def test_disable_validation(self):
        for validate in TestUtils._validators:
            if validate is validate_generic:
                arg = 'string'
                var = 0
            elif validate is validate_levels:
                arg = (validate_generic(None, 'string'),)
                var = 0
            elif validate is validate_numeric:
                arg = 'integer',
                var = ''
            else:
                raise ValueError('Unknown validator: {!r}'
                                 .format(validate.__name__))

            self._test_disable_validation(validate, arg, var)

    def test_require_decoration(self):
        for validate in TestUtils._validators:
            if validate is validate_generic:
                arg = 'string'
                var = ''
            elif validate is validate_levels:
                arg = (validate_generic(None, 'string'),)
                var = ''
            elif validate is validate_numeric:
                arg = 'integer',
                var = 0
            else:
                raise ValueError('Unknown validator: {!r}'
                                 .format(validate.__name__))

            self._test_require_decoration(validate, arg, var)

    def test_validate_once(self):
        for validate in TestUtils._validators:
            if validate is validate_generic:
                arg = 'string'
                vars_ = ('', 0)
            elif validate is validate_levels:
                arg = (validate_generic(None, 'string'),)
                vars_ = ('', 0)
            elif validate is validate_numeric:
                arg = 'integer',
                vars_ = (0, '')
            else:
                raise ValueError('Unknown validator: {!r}'
                                 .format(validate.__name__))

            self._test_validate_once(validate, arg, vars_)

    def _test_disable_validation(self, validate, arg, var):
        def example(var):
            @decorate_validation
            def validate_input():
                validate('var', arg)

            validate_input()

        try:
            example(var)
        except TypeError:
            pass
        else:
            self.fail('{!r} should fail here.'.format(validate.__name__))

        try:
            disable_validation()
            example(var)
        except TypeError as e:
            print('\nException caught:\n    {}'.format(e.args[0]))
            self.fail('{!r} should disable validation.'
                      .format(disable_validation.__name__))
        finally:
            magni.utils.validation._util._disabled = False

    def _test_require_decoration(self, validate, arg, var):
        def example_with_decoration(var):
            @decorate_validation
            def validate_input():
                validate('var', arg)

            validate_input()

        def example_without_decoration(var):
            def validate_input():
                validate('var', arg)

            validate_input()

        try:
            example_with_decoration(var)
        except RuntimeError as e:
            print('\nException caught:\n    {}'.format(e.args[0]))
            self.fail('{!r} should enable validation.'
                      .format(decorate_validation.__name__))

        try:
            example_without_decoration(var)
        except RuntimeError:
            pass
        else:
            self.fail('{!r} should require decoration.'
                      .format(validate.__name__))

    def _test_validate_once(self, validate, arg, vars_):
        @validate_once
        def example(var):
            @decorate_validation
            def validate_input():
                validate('var', arg)

            validate_input()

        try:
            example(vars_[0])
            example(vars_[1])
        except TypeError:
            pass
        else:
            self.fail('{!r} should fail here with {!r} disabled.'
                      .format(validate.__name__, _validate_once.__name__))

        try:
            enable_validate_once()
            example(vars_[0])
            example(vars_[1])
        except TypeError as e:
            print('\nException caught:\n    {}'.format(e.args[0]))
            self.fail('{!r} should enable validation once.'
                      .format(enable_validate_once.__name__))
        finally:
            magni.utils.validation._util._validate_once_enabled = False
