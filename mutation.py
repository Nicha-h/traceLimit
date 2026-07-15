try:
    import libcst as cst
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "libcst", "-q"])
    import libcst as cst


class BugTransformer(cst.CSTTransformer):
    def __init__(self, target_fn: str, bug_type: str):
        self.target_fn = target_fn
        self.bug_type = bug_type
        self.in_target = False
        self._a_mutated = False
        self._b_mutated = False
        self._d_mutated = False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if node.name.value == self.target_fn:
            self.in_target = True
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if original_node.name.value == self.target_fn:
            self.in_target = False
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.CSTNode:
        if not self.in_target:
            return updated_node

        # Type A: subtract 1 from the last positional arg of the first range() call
        if self.bug_type == "A" and not self._a_mutated and isinstance(updated_node.func, cst.Name) and updated_node.func.value == "range":
            if updated_node.args:
                last_idx = len(updated_node.args) - 1
                orig_arg = updated_node.args[last_idx].value
                new_arg = cst.BinaryOperation(
                    left=orig_arg,
                    operator=cst.Minus(),
                    right=cst.Integer(value="1"),
                )
                new_args = list(updated_node.args)
                new_args[last_idx] = updated_node.args[last_idx].with_changes(value=new_arg)
                self._a_mutated = True
                return updated_node.with_changes(args=new_args)

        # Type D: swap first two positional Name/Attribute args in the first qualifying call
        if self.bug_type == "D" and not self._d_mutated:
            positional = [
                (i, arg)
                for i, arg in enumerate(updated_node.args)
                if arg.keyword is None
                and isinstance(arg.value, (cst.Name, cst.Attribute))
                and not (isinstance(arg.value, cst.Name) and arg.value.value == "None")
            ]
            if len(positional) >= 2:
                i0, arg0 = positional[0]
                i1, arg1 = positional[1]
                new_args = list(updated_node.args)
                new_args[i0] = arg0.with_changes(value=arg1.value)
                new_args[i1] = arg1.with_changes(value=arg0.value)
                self._d_mutated = True
                return updated_node.with_changes(args=new_args)

        return updated_node

    def leave_If(self, original_node: cst.If, updated_node: cst.If) -> cst.CSTNode:
        # Type B: flip the first if-condition inside the target function
        if self.in_target and self.bug_type == "B" and not self._b_mutated:
            test = updated_node.test
            if isinstance(test, cst.UnaryOperation) and isinstance(test.operator, cst.Not):
                # `if not x:` → `if x:`
                new_test = test.expression
            else:
                # `if x:` → `if not (x):`
                new_test = cst.UnaryOperation(
                    operator=cst.Not(whitespace_after=cst.SimpleWhitespace(" ")),
                    expression=test.with_changes(
                        lpar=[cst.LeftParen()],
                        rpar=[cst.RightParen()],
                    ),
                )
            self._b_mutated = True
            return updated_node.with_changes(test=new_test)
        return updated_node

    def leave_Comparison(self, original_node: cst.Comparison, updated_node: cst.Comparison) -> cst.CSTNode:
        if not self.in_target:
            return updated_node

        # Type C: flip == ↔ != on every comparison in the target function.
        # libcst Comparison.comparisons is Sequence[ComparisonTarget]; each has .operator.
        if self.bug_type == "C":
            new_comparisons = []
            for comp_target in updated_node.comparisons:
                if isinstance(comp_target.operator, cst.Equal):
                    new_comparisons.append(comp_target.with_changes(operator=cst.NotEqual()))
                elif isinstance(comp_target.operator, cst.NotEqual):
                    new_comparisons.append(comp_target.with_changes(operator=cst.Equal()))
                else:
                    new_comparisons.append(comp_target)
            return updated_node.with_changes(comparisons=new_comparisons)

        # Type D fallback: swap left/right of the first non-None, non-symmetric binary comparison.
        # Skip is/is not — swapping their operands is semantically identical.
        if self.bug_type == "D" and not self._d_mutated and len(updated_node.comparisons) == 1:
            left = updated_node.left
            comp = updated_node.comparisons[0]
            right = comp.comparator
            left_is_none = isinstance(left, cst.Name) and left.value == "None"
            right_is_none = isinstance(right, cst.Name) and right.value == "None"
            is_symmetric = isinstance(comp.operator, (cst.Is, cst.IsNot, cst.In, cst.NotIn))
            if not left_is_none and not right_is_none and not is_symmetric:
                self._d_mutated = True
                return updated_node.with_changes(
                    left=right,
                    comparisons=[comp.with_changes(comparator=left)],
                )

        return updated_node


def apply_mutation(file_content: str, target_fn: str, bug_type: str) -> str:
    try:
        source_tree = cst.parse_module(file_content)
        transformer = BugTransformer(target_fn, bug_type)
        modified_tree = source_tree.visit(transformer)
        return modified_tree.code
    except Exception:
        return file_content
