import libcst as cst

class BugTransformer(cst.CSTTransformer):
    def __init__(self, target_fn: str, bug_type: str):
        self.target_fn = target_fn
        self.bug_type = bug_type
        self.in_target = False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if node.name.value == self.target_fn:
            self.in_target = True
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if original_node.name.value == self.target_fn:
            self.in_target = False
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.CSTNode:
        # Type A: Off-by-one bug mutation target loop bounds
        if self.in_target and self.bug_type == "A" and isinstance(updated_node.func, cst.Name) and updated_node.func.value == "range":
            if updated_node.args:
                orig_arg = updated_node.args[0].value
                new_arg = cst.BinaryOperation(
                    left=orig_arg,
                    operator=cst.Minus(),
                    right=cst.Integer(value="1")
                )
                return updated_node.with_changes(args=[cst.Arg(value=new_arg)])
        return updated_node

def apply_mutation(file_content: str, target_fn: str, bug_type: str) -> str:
    """
    Parses a module into a Concrete Syntax Tree and injects the specified bug type structural mutation.
    """
    try:
        source_tree = cst.parse_module(file_content)
        transformer = BugTransformer(target_fn, bug_type)
        modified_tree = source_tree.visit(transformer)
        return modified_tree.code
    except Exception:
        return file_content