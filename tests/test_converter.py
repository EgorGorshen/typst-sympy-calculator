import pytest
import sympy

from TypstConverter import (
    TypstMathConverter,  # замените на актуальный путь к вашему модулю
)


@pytest.fixture
def convertor():
    convertor = TypstMathConverter()
    (
        operator,
        relation_op,
        additive_op,
        mp_op,
        postfix_op,
        reduce_op,
        func,
        func_mat,
        constant,
    ) = convertor.get_decorators()

    @func()
    def convert_sin(x):
        return sympy.sin(x)

    @func_mat()
    def convert_mat(mat):
        return sympy.matrices.Matrix(mat)

    convertor.define_symbol_base("x")
    convertor.define_symbol_base("y")
    convertor.define_symbol_base("z")

    return convertor


def test_basic_expression(convertor):
    expr = convertor.sympy("1 + sin^2 1/2 + x + 1")
    typst = convertor.typst(sympy.simplify(expr))
    assert typst == "x + (sin(1/2))^2 + 2"


def test_nested_power_expression(convertor):
    expr = convertor.sympy("(x y)^y^(z+1)")
    typst = convertor.typst(sympy.simplify(expr))
    assert typst == "(x y)^y^(z + 1)"


def test_matrix_expression(convertor):
    expr = convertor.sympy("mat(x + y, 2; z, 4)")
    typst = convertor.typst(sympy.simplify(expr))
    assert typst == "mat(x + y, 2; z, 4)"


def test_custom_function_expression(convertor):
    convertor.define_function("f_1")
    expr = convertor.sympy("f_1^2(1) + f_1(1)")
    typst = convertor.typst(sympy.simplify(expr))
    assert typst == "(f_1(1) + 1) f_1(1)"


def test_multiplicative_expression(convertor):
    expr = convertor.sympy("x * y * z")
    typst = convertor.typst(expr)
    assert typst == "x y z"


def test_multiplicative_with_addition_expression(convertor):
    expr = convertor.sympy("(x + 1) * y * z")
    typst = convertor.typst(expr)
    assert typst == "y z (x + 1)"


def test_sqrt_expression(convertor):
    expr = convertor.sympy("(x + 1) * y^(1/2)")
    typst = convertor.typst(expr)
    assert typst == "sqrt(y) (x + 1)"


def test_absolute_value_expression(convertor):
    expr = convertor.sympy("|x|")
    typst = convertor.typst(expr)
    assert typst == "|x|"


def test_integral_expression(convertor):
    expr = convertor.sympy("integral_1^2 x^2 dif x")
    typst = convertor.typst(expr)
    assert typst == "integral_1^2 x^2 dif x"
