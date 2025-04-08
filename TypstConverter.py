import math
from functools import reduce, wraps
from typing import Callable

import sympy
from sympy.printing.str import StrPrinter

from TypstParser import TypstMathParser


class TypstMathPrinter(StrPrinter):
    def paren(self, expr):
        return "(" + self.doprint(expr) + ")" if expr.is_Add else self.doprint(expr)

    def _print_list(self, expr):
        expr = list(set([self.doprint(item) for item in expr]))
        # lst = [self.doprint(item) for item in lst]
        return "(" + ", ".join(expr) + ")"

    def _print_tuple(self, expr):
        return "(" + ", ".join([self.doprint(item) for item in expr]) + ")"

    def _print_dict(self, d):
        if len(d) == 0:
            return "nothing"
        elif len(d) == 1:
            k, v = d.popitem()
            return self.doprint(k) + " = " + self.doprint(v)
        else:
            return (
                "cases("
                + ", ".join(
                    [self.doprint(k) + " = " + self.doprint(v) for k, v in d.items()]
                )
                + ")"
            )

    def _print_Mul(self, expr):
        def mul(x, y):
            if x == "1":
                return y
            if x == "-1":
                return "-" + y
            return x + " " + y

        if len(expr.args) >= 2 and expr.args[0].is_Number:
            num = expr.args[0]
            x = expr.args[1]
            if x.is_Pow and x.args[1].is_Number and x.args[1] < 0:
                return (
                    (
                        reduce(
                            mul,
                            [self.paren(arg) for arg in [num] + list(expr.args[2:])],
                        )
                    )
                    + " / "
                    + self.paren(sympy.simplify(x**-1))
                )
            else:
                return reduce(mul, [self.paren(arg) for arg in expr.args])
        else:
            x = expr.args[0]
            if x.is_Pow and x.args[1].is_Number and x.args[1] < 0:
                return (
                    (
                        reduce(mul, [self.paren(arg) for arg in expr.args[1:]])
                        if len(expr.args) > 1
                        else "1"
                    )
                    + " / "
                    + self.paren(sympy.simplify(x**-1))
                )
            else:
                return reduce(mul, [self.paren(arg) for arg in expr.args])

    # matrix form: mat(1, 2; 3, 4)
    def _print_MatrixBase(self, expr):
        n, m, mat_flattern = expr.args
        res = "mat("
        rows = [mat_flattern[i : i + m] for i in range(0, n * m, m)]
        res += "; ".join(map(lambda row: ", ".join(map(self.doprint, row)), rows))
        res += ")"
        return res

    def _print_Limit(self, expr):
        e, z = expr.args
        return "lim_(%s -> %s) %s" % (self._print(z), self._print(z), self._print(e))

    def _print_Integral(self, expr):
        e, lims = expr.args
        if len(lims) > 1:
            return "integral_%s^%s %s dif %s" % (
                self.paren(lims[1]),
                self.paren(lims[2]),
                self._print(e),
                self._print(lims[0]),
            )
        else:
            return "integral %s dif %s" % (self._print(e), self._print(lims))

    def _print_Sum(self, expr):
        e, lims = expr.args
        return "sum_(%s = %s)^%s %s" % (
            self._print(lims[0]),
            self._print(lims[1]),
            self.paren(lims[2]),
            self._print(e),
        )

    def _print_Product(self, expr):
        e, lims = expr.args
        return "product_(%s = %s)^%s %s" % (
            self._print(lims[0]),
            self._print(lims[1]),
            self.paren(lims[2]),
            self._print(e),
        )

    def _print_factorial(self, expr):
        return "%s!" % self._print(expr.args[0])

    def _print_Derivative(self, expr):
        e = expr.args[0]
        wrt = expr.args[1]
        return "dif / (dif %s) (%s)" % (self._print(wrt), self._print(e))

    def _print_Abs(self, expr):
        return "|%s|" % self._print(expr.args[0])

    def _print_Equality(self, expr):
        return "%s = %s" % (self._print(expr.args[0]), self._print(expr.args[1]))

    # using ^ but not ** for power
    def _print_Pow(self, expr):
        # TODO: добавить использование root если степень у выражения дробная (1/n)
        b, e = expr.args
        base = self.doprint(b) if b.is_Atom else "(" + self.doprint(b) + ")"
        if e is sympy.S.Half:
            return "sqrt(%s)" % self.doprint(b)
        if -e is sympy.S.Half:
            return "1 / sqrt(%s)" % self.doprint(b)
        if e is -sympy.S.One:
            return "1 / %s" % base
        if e.is_Atom or e.is_Pow:
            return base + "^" + self.doprint(e)
        else:
            return base + "^" + "(" + self.doprint(e) + ")"

    def _print_And(self, expr):
        return " and ".join([self.doprint(item) for item in expr.args])

    def _print_Or(self, expr):
        return " or ".join([self.doprint(item) for item in expr.args])

    def _print_Not(self, expr):
        return "not " + self.doprint(expr.args[0])


class TypstMathConverter(object):
    id2type = {}
    id2func = {}

    def __init__(self) -> None:
        self.parser = TypstMathParser()
        self.printer = TypstMathPrinter()

    def define(self, name: str, type_: str, func: Callable | None = None):
        """
        Register a new identifier (function, constant, or operator) with its type and optional implementation.

        WARNING:
            This method adds both versions of the identifier: with and without the leading '#' symbol,
            to ensure compatibility with Typst syntax variations. The type of the identifier is stored
            in `id2type`, and its implementation (if provided) is stored in `id2func`.

        :param name (str): The name of the identifier to register. Can optionally start with '#'.
        :param type_ (str): The type of the identifier (e.g., 'func', 'const', 'diff').
        :param func (Callable | None): Optional function implementation associated with this identifier.
        """

        name = name.lstrip("#")
        self.id2type[name.split("_")[0]] = type_
        self.id2type["#" + name.split("_")[0]] = type_

        if func is not None:
            self.id2func[name] = func
            self.id2func["#" + name] = func

    def undefine(self, name: str):
        """
        Remove the identifier and its function (if any) from the registry,
        handling both '#' and non-'#' versions.


        :param name (str): The name of the identifier to remove. Can start with '#'.
        """
        base = name.lstrip("#")
        for key in (base, f"#{base}"):
            self.id2type.pop(key.split("_")[0], None)
            self.id2func.pop(key, None)

    def define_accent(self, accent_name: str):
        """
        Register an accent operator (e.g., vector, hat) by name.
        """
        self.define(accent_name, "ACCENT_OP")

    def define_symbol_base(self, symbol_base_name: str):
        """
        Register a base symbol (e.g., Latin/Greek letter used as a variable).
        """
        self.define(symbol_base_name, "SYMBOL_BASE")

    def define_function(self, function_name: str):
        """
        Register a mathematical function (e.g., sin, log, f).
        """
        self.define(function_name, "FUNC")

    def parse(self, typst_math: str):
        self.parser.id2type = self.id2type
        return self.parser.parse(typst_math)

    def sympy(self, typst_math: str):
        math = self.parse(typst_math)
        return self.convert_relation(math.relation())

    def typst(self, sympy_expr) -> str:
        return self.printer.doprint(sympy_expr)

    def convert_relation(self, relation):
        """
        Converts a parser node 'relation' into a Sympy object representing a logical expression (Eq, Ne, Lt, etc.).

        :param relation: A parser node describing the relation.
        :return: A Sympy expression (sympy.Expr), for instance sympy.Eq(x, y).
        """

        relation_op = relation.RELATION_OP()

        if not relation_op:
            expr = relation.expr()
            assert expr
            return self.convert_expr(expr)

        relations = relation.relation()
        assert len(relations) == 2
        op = relation_op.getText()

        standard_ops = {
            "=": sympy.Eq,
            "==": sympy.Eq,
            "!=": sympy.Ne,
            "<": sympy.Lt,
            ">": sympy.Gt,
            "<=": sympy.Le,
            ">=": sympy.Ge,
        }

        if op in standard_ops:
            return standard_ops[op](
                self.convert_relation(relations[0]), self.convert_relation(relations[1])
            )

        if op in self.id2type and self.id2type[op] == "RELATION_OP":
            assert op in self.id2func, f"function for {op} not found"
            return self.id2func[op](relation)

        raise Exception(f"unknown relation operator {op}")

    def convert_expr(self, expr):
        return self.convert_additive(expr.additive())

    def convert_additive(self, additive):
        """
        Converts an additive expression into a SymPy expression.

        Handles standard operations like '+' and '-', as well as user-defined additive operators
        registered in 'id2type' and 'id2func'. Falls back to a multiplicative expression if no
        additive operator is present.

        :param additive: Parser node representing an additive expression (e.g., a + b, a - b).
        :return: A SymPy expression representing the result of the additive operation.
        :raises Exception: If the operator is unknown or unregistered.
        """
        additive_op = additive.ADDITIVE_OP()

        if not additive_op:
            return self.convert_mp(additive.mp())

        additives = additive.additive()
        assert len(additives) == 2

        op = additive_op.getText()

        standard_ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b}

        if op in standard_ops:
            return standard_ops[op](
                self.convert_additive(additives[0]), self.convert_additive(additives[1])
            )

        if op in self.id2type and self.id2type[op] == "ADDITIVE_OP":
            assert op in self.id2func, f"function for {op} not found"
            return self.id2func[op](additive)

        raise Exception(f"unknown additive operator {op}")

    def convert_mp(self, mp, is_denominator=False):
        """
        Converts a parser node 'mp' (multiplicative expression) into a computed value or a Sympy object.

        :param mp: A parser node describing a multiplicative expression (e.g. a * b, a / b).
        :param is_denominator: A boolean indicating whether the expression is in the denominator (used for context-specific behavior).
        :return: The result of evaluating the multiplicative expression.
        :raises Exception: If the operator is unknown and not registered in 'id2type' and 'id2func'.
        """

        mp_op = mp.MP_OP()

        if not mp_op:
            return self.convert_unary(mp.unary(), is_denominator=is_denominator)

        mps = mp.mp()
        assert len(mps) == 2
        op = mp_op.getText()

        def mp_at(i, is_denominator=False):
            return self.convert_mp(mps[i], is_denominator=is_denominator)

        if op == "*":
            return mp_at(0) * mp_at(1)

        if op == "/":
            return mp_at(0) / mp_at(1, True)

        if op == "\\/":
            return mp_at(0) / mp_at(1, True)

        if op in self.id2type and self.id2type[op] == "MP_OP":
            assert op in self.id2func, f"function for {op} not found"
            return self.id2func[op](mp)

        raise Exception(f"unknown mp operator {op}")

    def convert_unary(self, unary, is_denominator=False):
        """
        Converts a parser node 'unary' into a computed value or a SymPy object.

        :param unary: A parser node representing a unary expression, which may include a leading '+' or '-' operator,
                      or a sequence of postfix expressions.
        :param is_denominator: A boolean indicating whether the expression is part of a denominator (affects evaluation order).
        :return: The result of evaluating the unary expression, as a numeric value or a SymPy object.
        :raises Exception: If the unary operator is unsupported.
        """

        additive_op = unary.ADDITIVE_OP()

        if not additive_op:
            postfixes = [self.convert_postfix(pos) for pos in unary.postfix()]
            assert len(postfixes) != 0

            if not is_denominator:
                return math.prod(postfixes)

            return postfixes[0] / math.prod(postfixes[1:])

        unary = unary.unary()
        assert unary

        op = additive_op.getText()

        if op == "+":
            return self.convert_unary(unary, is_denominator=is_denominator)
        elif op == "-":
            return -self.convert_unary(unary, is_denominator=is_denominator)

        raise Exception(f"unsupport unary operator {op}")

    def convert_eval_at(self, expr, eval_at):
        """
        Evaluates a symbolic expression at a given point or over a range, using eval bar notation.

        :param expr: SymPy expression to evaluate.
        :param eval_at: Parser node representing evaluation bounds (e.g. |_{x=1}^{2}).
        :return: expr(sup) - expr(sub), or expr(val) if only one bound is provided.
        :raises ValueError: If no bounds are provided.
        """
        symbol, sub, sup = self.convert_subsupassign(eval_at.subsupassign())
        symbol = symbol or next(iter(expr.free_symbols), None)

        if symbol is None:
            raise ValueError("Cannot determine the symbol to evaluate at.")

        if sub is None and sup is None:
            raise ValueError("At least one of sub or sup must be specified.")

        if sub is None or sup is None:
            return expr.subs(symbol, sub or sup)

        return expr.subs(symbol, sup) - expr.subs(symbol, sub)

    def convert_postfix(self, postfix):
        """
        Processes postfix operators (transpose, factorial, eval@, custom) for a parsed expression.

        :param postfix: ANTLR parse tree node containing:
                        - exp(): Base expression
                        - postfix_op(): List of postfix operations
        :return: sympy.Expr with applied postfix operations
        :raises ValueError: If base expression is missing
        :raises KeyError: For unregistered custom operators
        :raises RuntimeError: For unsupported postfix operations
        """
        if not (exp := postfix.exp()):
            raise ValueError("Postfix expression requires base (e.g., 'x^T')")

        result = self.convert_exp(exp)

        # Optimization: precompute list of postfix operators
        for op_node in postfix.postfix_op() or []:
            result = self._apply_postfix_op(result, op_node)

        return result

    def _apply_postfix_op(self, expr, op_node):
        """
        Applies a single postfix operator to an expression.

        :param expr: sympy.Expr to modify
        :param op_node: ANTLR operator node containing:
                        - eval_at()
                        - transpose()
                        - POSTFIX_OP()
        :return: Modified sympy.Expr
        :raises RuntimeError: For unrecognized operator nodes
        """
        if eval_at := op_node.eval_at():
            return self.convert_eval_at(expr, eval_at)
        if op_node.transpose():
            return sympy.transpose(expr)

        if not (op := op_node.POSTFIX_OP()):
            raise RuntimeError(f"Unsupported postfix node: {op_node.getText()}")

        return self._get_postfix_handler(op.getText())(expr)

    def _get_postfix_handler(self, op: str) -> Callable:
        """
        Retrieves handler function for a postfix operator.

        :param op: Operator symbol (e.g., '!', '%', custom)
        :return: Callable handler function taking (expr, op) arguments
        :raises KeyError: For unregistered custom operators
        :raises RuntimeError: For unknown operators
        """
        # Core operator handlers
        base_handlers = {
            "!": lambda e: sympy.factorial(e),
            "%": lambda e: e * 0.01,
        }

        # Custom operator resolution
        if handler := base_handlers.get(op):
            return handler
        if self.id2type.get(op) == "POSTFIX_OP":
            if op not in self.id2func:
                raise KeyError(f"Postfix operator '{op}' not registered")
            return lambda e: self.id2func[op](e)

        raise RuntimeError(f"Unrecognized postfix operator: '{op}'")

    def convert_exp(self, exp):
        """
        Converts an expression node with optional superscript into a SymPy expression.

        :param exp: Parser node representing a base expression, optionally raised to a power (e.g., a^b).
        :return: A SymPy expression representing the parsed mathematical expression.
        :raises AssertionError: If the base component (comp) is missing.
        """
        comp = exp.comp()
        assert comp
        if supexpr := exp.supexpr():
            return self.convert_comp(comp) ** self.convert_supexpr(supexpr)
        return self.convert_comp(comp)

    def convert_comp(self, comp):
        """
        Converts a component node into a SymPy expression by dispatching to the appropriate handler.

        The function checks the type of the component (e.g., group, function, matrix, etc.)
        and passes it to the corresponding converter method.

        :param comp: Parser node representing a mathematical component.
        :return: A SymPy expression matching the structure of the component.
        """
        converters = [
            ("group", self.convert_group),
            ("abs_group", self.convert_abs_group),
            ("func", self.convert_func),
            ("matrix", self.convert_matrix),
            ("reduceit", self.convert_reduceit),
            ("lim", self.convert_lim),
            ("log", self.convert_log),
            ("integral", self.convert_integral),
            ("atom", self.convert_atom),
        ]

        for attr, handler in converters:
            if value := getattr(comp, attr)():
                return handler(value)

    def convert_group(self, group):
        return self.convert_expr(group.expr())

    def convert_abs_group(self, abs_group):
        return sympy.Abs(self.convert_expr(abs_group.expr()))

    def convert_func(self, func):
        """
        Converts a function node from the parser into a Sympy expression.

        :param func: A parser node representing a function call.
        :return: A Sympy expression representing the function, optionally raised to a power.
        """
        func_base_name = func.FUNC().getText()
        subargs = func.subargs().getText() if func.subargs() else ""
        func_name = f"{func_base_name}{subargs}"

        supexpr = self.convert_supexpr(func.supexpr()) if func.supexpr() else None

        if not (self.id2type.get(func_base_name) == "FUNC"):
            raise ValueError(f"Unknown function type: {func_name}")

        args = (
            [self.convert_relation(arg) for arg in func.args().relation()]
            if func.args()
            else [self.convert_mp(func.mp())]
        )

        if registered_func := self.id2func.get(func_name):
            result = registered_func(func)
        else:
            result = sympy.Function(func_name)(*args)

        return result**supexpr if supexpr else result

    def convert_matrix(self, matrix):
        """
        Converts a matrix function node from the parser into a Sympy object,
        using a registered handler if available.

        :param matrix: A parser node representing a matrix function.
        :return: A Sympy object or result of a registered matrix function.
        :raises Exception: If the matrix function is not recognized or not registered.
        """
        func_name = matrix.FUNC_MAT().getText()
        if func_name in self.id2type and self.id2type[func_name] == "FUNC_MAT":
            assert func_name in self.id2func, f"function for {func_name} not found"
            return self.id2func[func_name](matrix)
        raise Exception(f"unknown matrix function {func_name}")

    def convert_subassign(self, subassign):
        """
        Converts a subassignment node from the parser into a SymPy expression or equality components.

        :param subassign: A parser node representing a subassignment (atom, expression, or relation).
        :return: A tuple (lhs, rhs) where lhs is None or a Symbol, and rhs is a SymPy expression.
        :raises AssertionError: If the converted relation is not a SymPy Equality.
        """
        if atom := subassign.atom():
            return None, self.convert_atom(atom)

        if expr := subassign.expr():
            return None, self.convert_expr(expr)

        if relation := subassign.relation():
            rel = self.convert_relation(relation)
            # NOTE: type analysis indicating unreachable???
            assert isinstance(rel, sympy.Equality)
            return rel.lhs, rel.rhs

    def convert_supassign(self, supassign):
        """
        Converts a supsassignment node from the parser into a SymPy expression or equality components,
        enforcing the left-hand side to be a Symbol.

        :param supsassign: A parser node representing a supsassignment (exp, expression, or relation).
        :return: A tuple (lhs, rhs) where lhs is None or a Symbol, and rhs is a SymPy expression.
        :raises AssertionError: If the relation is not a SymPy Equality or the lhs is not a Symbol.
        """
        if exp := supassign.exp():
            return None, self.convert_exp(exp)

        if expr := supassign.expr():
            return None, self.convert_expr(expr)

        if relation := supassign.relation():
            rel = self.convert_relation(relation)
            # NOTE: type analysis indicating unreachable???
            assert isinstance(rel, sympy.Equality)
            assert isinstance(rel.lhs, sympy.Symbol), (
                f"lhs of {supassign.relation().getText()} is not a symbol"
            )
            return rel.lhs, rel.rhs

    def convert_subsupassign(self, subsupassign):
        symbol = None
        sub = None
        sup = None
        # WARNING: according to the static analyzer, this code is useless.
        # because `self.convert_subassign` will always return either an error or None, some.
        # And here we use only the first variable (None)
        # NOTE: process sub
        if subassign := subsupassign.subassign():
            sym, sub = self.convert_subassign(subassign)
            if sym:
                symbol = sym

        # WARNING: according to the static analyzer, this code is useless.
        # because `self.convert_subassign` will always return either an error or None, some.
        # And here we use only the first variable (None)
        # NOTE: process sup
        if subsupassign.supexpr():
            sup = self.convert_supexpr(subsupassign.supexpr())
        elif subsupassign.supassign():
            sym, sup = self.convert_supassign(subsupassign.supassign())
            if sym:
                symbol = sym

        return (symbol, sub, sup)

    def convert_reduceit(self, reduceit):
        reduce_name = reduceit.REDUCE_OP().getText()
        if reduce_name in self.id2type and self.id2type[reduce_name] == "REDUCE_OP":
            assert reduce_name in self.id2func, f"function for {reduce_name} not found"
            return self.id2func[reduce_name](reduceit)
        else:
            raise Exception(f"unknown reduce function {reduce_name}")

    def convert_lim(self, lim):
        symbol = self.convert_symbol(lim.symbol())
        expr = self.convert_expr(lim.expr())
        additive = self.convert_additive(lim.additive())
        return sympy.Limit(additive, symbol, expr)

    def convert_log(self, log):
        if log.expr():
            value = self.convert_expr(log.expr())
        else:
            assert log.mp()
            value = self.convert_mp(log.mp())
        if log.subexpr():
            subexpr = self.convert_subexpr(log.subexpr())
            return sympy.log(value, subexpr)
        else:
            return sympy.log(value)

    def convert_integral(self, integral):
        subsupexpr = integral.subsupexpr()
        additive = self.convert_additive(integral.additive())
        symbol = self.convert_symbol(integral.symbol())
        if subsupexpr:
            subexpr, supexpr = self.convert_subsupexpr(subsupexpr)
            return sympy.Integral(additive, (symbol, subexpr, supexpr))
        else:
            return sympy.Integral(additive, symbol)

    def convert_subsupexpr(self, subsupexpr):
        subexpr = self.convert_subexpr(subsupexpr.subexpr())
        supexpr = self.convert_supexpr(subsupexpr.supexpr())
        return subexpr, supexpr

    def convert_subexpr(self, subexpr):
        if atom := subexpr.atom():
            return self.convert_atom(atom)
        if expr := subexpr.expr():
            return self.convert_expr(expr)
        raise Exception(f"unknown subexpr {subexpr.getText()}")

    def convert_supexpr(self, supexpr):
        if exp := supexpr.exp():
            return self.convert_exp(exp)
        if expr := supexpr.expr():
            return self.convert_expr(expr)
        raise Exception(f"unknown supexpr {supexpr.getText()}")

    def convert_atom(self, atom):
        if num := atom.NUMBER():
            return sympy.Rational(num.getText())
        if sym := atom.symbol():
            return self.convert_symbol(sym)

        raise Exception(f"unknown atom {atom.getText()}")

    def convert_symbol(self, symbol):
        symbol_name = symbol.getText()
        if symbol_name in self.id2func:
            # it is a constant function but not a symbol
            return self.id2func[symbol_name]()
        else:
            return sympy.Symbol(symbol_name)

    def get_decorators(env):
        class operator(object):
            def __init__(
                self, type: str, convert_ast: Callable, name: str = None, ast=False
            ):
                self.type = type
                self.convert_ast = convert_ast
                self.name = name
                self.func = None
                self.ast = ast
                self.env = env

            def __call__(self, func):
                assert isinstance(func, Callable)
                if self.name is None:
                    name = func.__name__
                    assert name.startswith("convert_"), (
                        f'function name "{name}" should start with "convert_"'
                    )
                    assert len(name) > len("convert_")
                    self.name = name[len("convert_") :].replace("_dot_", ".")
                if self.ast:
                    self.func = func
                else:
                    # convert ast to args and kwargs
                    @wraps(func)
                    def ast_func(*args, **kwargs):
                        args, kwargs = self.convert_ast(*args, **kwargs)
                        return func(*args, **kwargs)

                    self.func = ast_func
                    # save to env
                    self.env.define(self.name, self.type, self.func)
                return self.func

            def __repr__(self):
                return f"{self.type}(name = {self.name}, ast = {self.ast})"

        class relation_op(operator):
            def __init__(self, name: str = None, ast=False):
                def convert_ast(relation):
                    return [
                        self.env.convert_relation(relation)
                        for relation in relation.relation()
                    ], {}

                super().__init__("RELATION_OP", convert_ast, name, ast)

        class additive_op(operator):
            def __init__(self, name: str = None, ast=False):
                def convert_ast(additive):
                    return [
                        self.env.convert_additive(additive)
                        for additive in additive.additive()
                    ], {}

                super().__init__("ADDITIVE_OP", convert_ast, name, ast)

        class mp_op(operator):
            def __init__(self, name: str = None, ast=False):
                def convert_ast(mp):
                    return [self.env.convert_mp(mp) for mp in mp.mp()], {}

                super().__init__("MP_OP", convert_ast, name, ast)

        class postfix_op(operator):
            def __init__(self, name: str = None, ast=False):
                # unsupported ast so do nothing
                def convert_ast(result):
                    return [result], {}

                super().__init__("POSTFIX_OP", convert_ast, name, ast)

        class reduce_op(operator):
            def __init__(self, name: str = None, ast=False):
                def convert_ast(reduceit):
                    # reduceit: REDUCE_OP subsupassign mp;
                    symbol, sub, sup = self.env.convert_subsupassign(
                        reduceit.subsupassign()
                    )
                    assert sub is not None and sup is not None
                    mp = self.env.convert_mp(reduceit.mp())
                    if symbol is None:
                        # get the first symbol in mp
                        symbol = mp.free_symbols.pop()
                    return [mp, (symbol, sub, sup)], {}

                super().__init__("REDUCE_OP", convert_ast, name, ast)

        class func(operator):
            def __init__(self, name: str = None, ast=False):
                def convert_ast(func):
                    func_args = func.args()
                    if func_args:
                        args = [
                            self.env.convert_relation(arg)
                            for arg in func_args.relation()
                        ]
                    else:
                        args = [self.env.convert_mp(func.mp())]
                    return args, {}

                super().__init__("FUNC", convert_ast, name, ast)

        class func_mat(operator):
            def __init__(self, name: str = None, ast=False):
                def convert_ast(matrix):
                    mat = [
                        [self.env.convert_relation(arg) for arg in args.relation()]
                        for args in matrix.mat_args().args()
                    ]
                    return [mat], {}

                super().__init__("FUNC_MAT", convert_ast, name, ast)

        class constant:
            def __init__(self, name: str = None, ast=False):
                self.type = "CONSTANT"
                self.name = name
                self.func = None
                self.env = env

            def __call__(self, func):
                assert isinstance(func, Callable)
                if self.name is None:
                    name = func.__name__
                    assert name.startswith("convert_"), (
                        f'function name "{name}" should start with "convert_"'
                    )
                    assert len(name) > len("convert_")
                    self.name = name[len("convert_") :].replace("_dot_", ".")
                self.func = func
                self.env.define_symbol_base(self.name.split("_")[0])
                self.env.id2func[self.name] = self.func
                return self.func

            def __repr__(self):
                return f"{self.type}(name = {self.name})"

        return (
            operator,
            relation_op,
            additive_op,
            mp_op,
            postfix_op,
            reduce_op,
            func,
            func_mat,
            constant,
        )


if __name__ == "__main__":
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
    expr = convertor.sympy("1 + sin^2 1/2 + x + 1")
    typst = convertor.typst(sympy.simplify(expr))
    assert typst == "x + (sin(1/2))^2 + 2"
    print("PASS")

    expr = convertor.sympy("(x y)^y^(z+1)")
    typst = convertor.typst(sympy.simplify(expr))
    assert typst == "(x y)^y^(z + 1)"
    print("PASS")

    expr = convertor.sympy("mat(x + y, 2; z, 4 - 1)")
    typst = convertor.typst(sympy.simplify(expr))
    assert typst == "mat(x + y, 2; z, 3)"
    print("PASS")

    convertor.define_function("f_1")
    expr = convertor.sympy("f_1^2(1) + f_1(1)")
    typst = convertor.typst(sympy.simplify(expr))
    assert typst == "(f_1(1) + 1) f_1(1)"
    print("PASS")

    expr = convertor.sympy("x * y * z")
    typst = convertor.typst(expr)
    assert typst == "x y z"
    print("PASS")

    expr = convertor.sympy("(x + 1) * y * z")
    typst = convertor.typst(expr)
    assert typst == "y z (x + 1)"
    print("PASS")

    expr = convertor.sympy("(x + 1) * y^(1/2)")
    typst = convertor.typst(expr)
    assert typst == "sqrt(y) (x + 1)"
    print("PASS")

    expr = convertor.sympy("|x|")
    typst = convertor.typst(expr)
    assert typst == "|x|"
    print("PASS")

    expr = convertor.sympy("integral_1^2 x^2 dif x")
    typst = convertor.typst(expr)
    assert typst == "integral_1^2 x^2 dif x"
    print("PASS")
