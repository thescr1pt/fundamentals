from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import re
import uvicorn

app = FastAPI(
    title="Compiler Analyzer API",
    description="API for processing code through lexical, syntax, semantic, and ICG phases",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Lexical Analyzer ---
token_specification = [
    ("NUMBER", r'\d+(\.\d+)?'),
    ("ASSIGN", r'='),
    ("ID", r'[A-Za-z_]\w*'),
    ("OP", r'[+\-*/]'),
    ("LPAREN", r'\('),
    ("RPAREN", r'\)'),
    ("SKIP", r'[ \t\n]+'),
    ("MISMATCH", r'.'),
]
tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)


def lexer(code):
    tokens, symbol_table, id_counter = [], {}, 1
    prev_token_kind = None

    for mo in re.finditer(tok_regex, code):
        kind, value = mo.lastgroup, mo.group()

        if prev_token_kind == "NUMBER" and kind == "ID":
            raise RuntimeError(
                f"Syntax error: number cannot be directly followed by identifier '{value}' without an operator")

        if kind == "ID":
            if value not in symbol_table:
                symbol_table[value] = f'id{id_counter}'
                id_counter += 1
            tokens.append(('ID', symbol_table[value], value))
            prev_token_kind = "ID"
        elif kind in ["NUMBER", "ASSIGN", "OP", "LPAREN", "RPAREN"]:
            tokens.append((kind, value))
            prev_token_kind = kind
        elif kind == "SKIP":
            continue
        else:
            raise RuntimeError(f"Unexpected character {value!r}")
    return tokens, symbol_table


# --- Hybrid Lexer (uses V1, V2 instead of id1, id2, and "is" instead of "=") ---
def hybrid_lexer(code):
    tokens, symbol_table, id_counter = [], {}, 1
    prev_token_kind = None

    for mo in re.finditer(tok_regex, code):
        kind, value = mo.lastgroup, mo.group()

        if prev_token_kind == "NUMBER" and kind == "ID":
            raise RuntimeError(
                f"Syntax error: number cannot be directly followed by identifier '{value}' without an operator")

        if kind == "ID":
            if value not in symbol_table:
                symbol_table[value] = f'V{id_counter}'
                id_counter += 1
            tokens.append(('ID', symbol_table[value], value))
            prev_token_kind = "ID"
        elif kind == "ASSIGN":
            tokens.append(('ASSIGN', 'is'))
            prev_token_kind = kind
        elif kind in ["NUMBER", "OP", "LPAREN", "RPAREN"]:
            tokens.append((kind, value))
            prev_token_kind = kind
        elif kind == "SKIP":
            continue
        else:
            raise RuntimeError(f"Unexpected character {value!r}")
    return tokens, symbol_table


# --- Syntax Analyzer ---
class Node:
    def __init__(self, value, left=None, right=None, node_type="OP", original_name=None):
        self.value, self.left, self.right = value, left, right
        self.node_type, self.original_name = node_type, original_name
        self.type_info = None

    def to_dict(self):
        """Convert node to dictionary for JSON serialization"""
        return {
            'value': self.value,
            'node_type': self.node_type,
            'type_info': self.type_info,
            'left': self.left.to_dict() if self.left else None,
            'right': self.right.to_dict() if self.right else None
        }


class Parser:
    def __init__(self, tokens):
        self.tokens, self.pos = tokens, 0

    def current_token(self):
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            # Ensure we always return a 3-tuple
            if len(token) == 2:
                return (token[0], token[1], token[1])
            return token
        return (None, None, None)

    def eat(self, expected_kind):
        token = self.current_token()
        if token[0] == expected_kind:
            self.pos += 1
            return token
        raise SyntaxError(f"Expected {expected_kind} but found {token[0]}")

    def parse(self):
        return self.statement()

    def statement(self):
        id_token = self.eat('ID')
        left = Node(id_token[1], node_type='ID', original_name=id_token[2])
        op_token = self.eat('ASSIGN')
        right = self.expression()
        return Node(op_token[1], left, right, node_type='ASSIGN')

    def expression(self):
        node = self.term()
        token = self.current_token()
        while token[0] == 'OP' and token[1] is not None and token[1] in '+-':
            op = self.eat('OP')
            node = Node(op[1], node, self.term())
            token = self.current_token()
        return node

    def term(self):
        node = self.factor()
        token = self.current_token()
        while token[0] == 'OP' and token[1] is not None and token[1] in '*/':
            op = self.eat('OP')
            node = Node(op[1], node, self.factor())
            token = self.current_token()
        return node

    def factor(self):
        kind, value, *_ = self.current_token()
        if kind == 'NUMBER':
            self.eat('NUMBER')
            return Node(value, node_type='NUMBER')
        if kind == 'ID':
            id_token = self.eat('ID')
            return Node(id_token[1], node_type='ID', original_name=id_token[2])
        if kind == 'LPAREN':
            self.eat('LPAREN')
            node = self.expression()
            self.eat('RPAREN')
            return node
        raise SyntaxError(f"Unexpected token {kind}")


# --- Semantic Analyzer ---
def get_type(node, type_table):
    if node.type_info == "int to float": return 'float'
    if node.node_type == 'NUMBER': return 'float' if '.' in str(node.value) else 'int'
    if node.node_type == 'ID': return type_table.get(node.original_name, 'int')
    if node.node_type in ['OP', 'ASSIGN'] and node.left and node.right:
        is_float = get_type(node.left, type_table) == 'float' or get_type(node.right, type_table) == 'float'
        return 'float' if is_float else 'int'
    return None


def mark_leaves_for_coercion(node):
    if not node: return
    if node.node_type in ['ID', 'NUMBER'] and not (node.node_type == 'NUMBER' and '.' in node.value):
        node.type_info = "int to float"
    else:
        mark_leaves_for_coercion(node.left)
        mark_leaves_for_coercion(node.right)


def semantic_analysis(node, type_table):
    if not node: return
    semantic_analysis(node.left, type_table)
    semantic_analysis(node.right, type_table)
    if node.node_type in ["OP", "ASSIGN"] and node.left and node.right:
        left_type, right_type = get_type(node.left, type_table), get_type(node.right, type_table)
        if left_type != right_type:
            if left_type == 'int': mark_leaves_for_coercion(node.left)
            if right_type == 'int': mark_leaves_for_coercion(node.right)


# --- Direct Execution (Hybrid Approach Phase 4) ---
def direct_execution(node, value_table, type_table):
    """
    Evaluate the tree and annotate each node with its computed value.
    Returns the computed value of the node.
    """
    if not node:
        return None
    
    if node.node_type == 'NUMBER':
        val = float(node.value)
        node.computed_value = val
        return val
    
    if node.node_type == 'ID':
        # Get value from value_table using original_name
        val = value_table.get(node.original_name, 0)
        # Convert to float if marked for coercion
        if node.type_info == "int to float":
            val = float(val)
        node.computed_value = val
        return val
    
    if node.node_type in ['OP', 'ASSIGN']:
        left_val = direct_execution(node.left, value_table, type_table)
        right_val = direct_execution(node.right, value_table, type_table)
        
        op = node.value
        if op == '+':
            result = left_val + right_val
        elif op == '-':
            result = left_val - right_val
        elif op == '*':
            result = left_val * right_val
        elif op == '/':
            result = left_val / right_val if right_val != 0 else 0
        elif op == 'is' or op == '=':
            result = right_val
        else:
            result = 0
        
        node.computed_value = result
        return result
    
    return None


class NodeWithExecution:
    """Extended Node class that includes computed values for JSON serialization"""
    @staticmethod
    def to_dict_with_execution(node):
        if not node:
            return None
        
        result = {
            'value': node.value,
            'node_type': node.node_type,
            'type_info': node.type_info,
            'computed_value': getattr(node, 'computed_value', None),
            'left': NodeWithExecution.to_dict_with_execution(node.left),
            'right': NodeWithExecution.to_dict_with_execution(node.right)
        }
        return result


# --- Intermediate Code Generator ---
def collect_conversions(node, type_table, conversions, temp_counter, skip_left_assign=False):
    if not node:
        return temp_counter

    if skip_left_assign and node.node_type == 'ASSIGN':
        temp_counter = collect_conversions(node.right, type_table, conversions, temp_counter, False)
        return temp_counter

    if node.node_type in ['ID', 'NUMBER'] and node.type_info == "int to float":
        if node.node_type == 'ID':
            key = ('ID', node.original_name)
            if key not in conversions:
                temp_name = f"temp{temp_counter}"
                temp_counter += 1
                # Use symbol table ID (id1, id2, etc.) in the instruction
                symbol_id = node.value
                conversions[key] = (temp_name, f"{temp_name} = float({symbol_id})")
        elif node.node_type == 'NUMBER':
            key = ('NUMBER', node.value)
            if key not in conversions:
                temp_name = f"temp{temp_counter}"
                temp_counter += 1
                conversions[key] = (temp_name, f"{temp_name} = float({node.value})")

    temp_counter = collect_conversions(node.left, type_table, conversions, temp_counter, False)
    temp_counter = collect_conversions(node.right, type_table, conversions, temp_counter, False)

    return temp_counter


def generate_icg(node, type_table, instructions, temp_counter, conversions=None):
    if not node:
        return None, temp_counter

    if node.node_type == 'ID':
        # Use the symbol table ID (e.g., id1, id2) instead of original name
        symbol_id = node.value  # This is the id1, id2, etc.
        if node.type_info == "int to float":
            key = ('ID', node.original_name)
            if conversions and key in conversions:
                return conversions[key][0], temp_counter
        return symbol_id, temp_counter

    elif node.node_type == 'NUMBER':
        if node.type_info == "int to float":
            key = ('NUMBER', node.value)
            if conversions and key in conversions:
                return conversions[key][0], temp_counter
        return node.value, temp_counter

    elif node.node_type == 'ASSIGN':
        right_result, temp_counter = generate_icg(node.right, type_table, instructions, temp_counter, conversions)
        # Use symbol table ID (id1, id2, etc.) instead of original name
        symbol_id = node.left.value
        instructions.append(f"{symbol_id} = {right_result}")
        return symbol_id, temp_counter

    elif node.node_type == 'OP':
        left_result, temp_counter = generate_icg(node.left, type_table, instructions, temp_counter, conversions)
        right_result, temp_counter = generate_icg(node.right, type_table, instructions, temp_counter, conversions)

        temp_name = f"temp{temp_counter}"
        temp_counter += 1
        instructions.append(f"{temp_name} = {left_result} {node.value} {right_result}")
        return temp_name, temp_counter

    return None, temp_counter


# --- Code Optimizer ---
def optimize_code(instructions, conversions):
    """
    Optimize intermediate code by:
    1. Eliminating conversion steps (float() calls)
    2. Replacing temp variables from conversions with their optimized values
    3. Keeping intermediate temp operations but with cleaner values
    """
    if not instructions:
        return []
    
    # Build mapping of conversion temps to their optimized values
    conversion_map = {}  # Maps temp variable to optimized value
    
    for key, (temp_name, instruction) in conversions.items():
        key_type, key_value = key
        if key_type == 'ID':
            # Extract the symbol ID from the instruction (e.g., temp1 = float(id2))
            # The instruction format is "tempX = float(idY)"
            import re as regex_module
            match = regex_module.search(r'float\(([^)]+)\)', instruction)
            if match:
                symbol_id = match.group(1)  # This will be id1, id2, etc.
                conversion_map[temp_name] = symbol_id
            else:
                conversion_map[temp_name] = key_value
        elif key_type == 'NUMBER':
            # temp1 = float(2) -> use '2.0'
            try:
                num_value = float(key_value)
                conversion_map[temp_name] = str(num_value)
            except ValueError:
                conversion_map[temp_name] = key_value
    
    # Filter out conversion instructions and replace conversion temps
    optimized = []
    
    for instruction in instructions:
        # Skip conversion instructions
        is_conversion = False
        for conv_temp in conversion_map:
            if instruction.startswith(f"{conv_temp} = float("):
                is_conversion = True
                break
        
        if is_conversion:
            continue
        
        # Replace conversion temps with their optimized values
        optimized_instruction = instruction
        for old_temp, new_value in conversion_map.items():
            optimized_instruction = optimized_instruction.replace(old_temp, new_value)
        
        optimized.append(optimized_instruction)
    
    # Now renumber temps consistently
    import re
    temp_map = {}
    temp_counter = 1
    renumbered = []
    
    for instruction in optimized:
        new_instruction = instruction
        # Find all temp variables in order they appear
        for match in re.finditer(r'temp\d+', instruction):
            temp_var = match.group()
            if temp_var not in temp_map:
                temp_map[temp_var] = f"temp{temp_counter}"
                temp_counter += 1
        
        # Replace all temps with their new numbers
        for old_temp, new_temp in temp_map.items():
            new_instruction = new_instruction.replace(old_temp, new_temp)
        
        renumbered.append(new_instruction)
    
    # Check if last instruction is "id = tempX" where tempX is ONLY a temp (no operations)
    if len(renumbered) >= 2:
        last_instr = renumbered[-1]
        if ' = ' in last_instr:
            left, right = last_instr.split(' = ', 1)
            left = left.strip()
            right = right.strip()
            
            # If it's just "id1 = temp3" (right side is ONLY a temp variable)
            if not left.startswith('temp') and re.match(r'^temp\d+$', right):
                # Find where this temp was assigned
                for i in range(len(renumbered) - 2, -1, -1):
                    if ' = ' in renumbered[i]:
                        prev_left, prev_right = renumbered[i].split(' = ', 1)
                        prev_left = prev_left.strip()
                        
                        if prev_left == right:
                            # Replace that line with the final assignment
                            renumbered[i] = f"{left} = {prev_right.strip()}"
                            # Remove the last line
                            renumbered.pop()
                            break
    
    return renumbered


# --- Code Generator ---
def generate_assembly(optimized_code, type_table):
    """
    Generate assembly code from optimized intermediate code.
    Uses float operations (LDF, MULF, ADDF, STRF) for float types
    and int operations (LD, MUL, ADD, STR) for int types.
    Only uses R1 and R2 registers.
    """
    if not optimized_code:
        return []
    
    import re
    assembly = []
    temp_registers = {}  # Maps temp variables to register numbers (R1 or R2)
    available_registers = [1, 2]  # Only R1 and R2 available
    next_register_idx = 0
    
    # Determine if we're dealing with floats (check if any variable is float or has decimal)
    use_float = False
    for var, var_type in type_table.items():
        if var_type == 'float':
            use_float = True
            break
    
    # Also check if any numbers in the code have decimals
    if not use_float:
        for instruction in optimized_code:
            if re.search(r'\d+\.\d+', instruction):
                use_float = True
                break
    
    # Set operation suffixes based on type
    suffix = 'F' if use_float else ''
    
    for instruction in optimized_code:
        if ' = ' not in instruction:
            continue
            
        left, right = instruction.split(' = ', 1)
        left = left.strip()
        right = right.strip()
        
        # Parse the right side to generate assembly
        if ' * ' in right:
            # Multiplication: temp1 = 2.0 * id2 or temp1 = id2 * 2.0
            parts = right.split(' * ')
            operand1 = parts[0].strip()
            operand2 = parts[1].strip()
            
            # Allocate register for this temp
            if left.startswith('temp'):
                if left not in temp_registers:
                    temp_registers[left] = available_registers[next_register_idx % 2]
                    next_register_idx += 1
                result_reg = f"R{temp_registers[left]}"
            else:
                result_reg = "R1"
            
            # Load the ID operand (never load constants)
            if operand1.startswith('id'):
                assembly.append(f"LD{suffix} {result_reg}, {operand1}")
                # Multiply by constant or another ID
                if operand2.startswith('id'):
                    assembly.append(f"MUL{suffix} {result_reg}, {result_reg}, {operand2}")
                else:
                    # operand2 is a constant
                    assembly.append(f"MUL{suffix} {result_reg}, {result_reg}, #{operand2}")
            elif operand2.startswith('id'):
                # operand1 is constant, operand2 is ID - load the ID
                assembly.append(f"LD{suffix} {result_reg}, {operand2}")
                assembly.append(f"MUL{suffix} {result_reg}, {result_reg}, #{operand1}")
            
            # Store if final variable
            if not left.startswith('temp'):
                assembly.append(f"STR{suffix} {left}, {result_reg}")
        
        elif ' + ' in right:
            # Addition: id1 = temp1 + temp2 or id1 = id2 + temp1
            parts = right.split(' + ')
            operand1 = parts[0].strip()
            operand2 = parts[1].strip()
            
            # Get register for first operand
            if operand1.startswith('temp'):
                reg1 = f"R{temp_registers[operand1]}"
            elif operand1.startswith('id'):
                # Check if second operand uses R1, if so use R2
                if operand2.startswith('temp') and temp_registers.get(operand2) == 1:
                    reg1 = "R2"
                else:
                    reg1 = "R1"
                assembly.append(f"LD{suffix} {reg1}, {operand1}")
            else:
                # Constant - check if second operand uses R1
                if operand2.startswith('temp') and temp_registers.get(operand2) == 1:
                    reg1 = "R2"
                else:
                    reg1 = "R1"
                assembly.append(f"LD{suffix} {reg1}, #{operand1}")
            
            # Get register for second operand
            if operand2.startswith('temp'):
                reg2 = f"R{temp_registers[operand2]}"
            elif operand2.startswith('id'):
                # Use the register that's not being used by operand1
                reg2 = "R2" if reg1 == "R1" else "R1"
                assembly.append(f"LD{suffix} {reg2}, {operand2}")
            else:
                reg2 = "R2" if reg1 == "R1" else "R1"
                assembly.append(f"LD{suffix} {reg2}, #{operand2}")
            
            # Add into first register
            assembly.append(f"ADD{suffix} {reg1}, {reg1}, {reg2}")
            
            # Store if final variable
            if not left.startswith('temp'):
                assembly.append(f"STR{suffix} {left}, {reg1}")
            elif left not in temp_registers:
                temp_registers[left] = int(reg1[1])  # Extract register number
        
        elif ' - ' in right:
            # Subtraction
            parts = right.split(' - ')
            operand1 = parts[0].strip()
            operand2 = parts[1].strip()
            
            if operand1.startswith('temp'):
                reg1 = f"R{temp_registers[operand1]}"
            elif operand1.startswith('id'):
                # Check if second operand uses R1, if so use R2
                if operand2.startswith('temp') and temp_registers.get(operand2) == 1:
                    reg1 = "R2"
                else:
                    reg1 = "R1"
                assembly.append(f"LD{suffix} {reg1}, {operand1}")
            else:
                if operand2.startswith('temp') and temp_registers.get(operand2) == 1:
                    reg1 = "R2"
                else:
                    reg1 = "R1"
                assembly.append(f"LD{suffix} {reg1}, #{operand1}")
            
            if operand2.startswith('temp'):
                reg2 = f"R{temp_registers[operand2]}"
            elif operand2.startswith('id'):
                reg2 = "R2" if reg1 == "R1" else "R1"
                assembly.append(f"LD{suffix} {reg2}, {operand2}")
            else:
                reg2 = "R2" if reg1 == "R1" else "R1"
                assembly.append(f"LD{suffix} {reg2}, #{operand2}")
            
            assembly.append(f"SUB{suffix} {reg1}, {reg1}, {reg2}")
            
            if not left.startswith('temp'):
                assembly.append(f"STR{suffix} {left}, {reg1}")
            elif left not in temp_registers:
                temp_registers[left] = int(reg1[1])
        
        elif ' / ' in right:
            # Division
            parts = right.split(' / ')
            operand1 = parts[0].strip()
            operand2 = parts[1].strip()
            
            if left.startswith('temp'):
                if left not in temp_registers:
                    temp_registers[left] = available_registers[next_register_idx % 2]
                    next_register_idx += 1
                result_reg = f"R{temp_registers[left]}"
            else:
                result_reg = "R1"
            
            if operand1.startswith('id'):
                assembly.append(f"LD{suffix} {result_reg}, {operand1}")
                if operand2.startswith('id'):
                    assembly.append(f"DIV{suffix} {result_reg}, {result_reg}, {operand2}")
                else:
                    assembly.append(f"DIV{suffix} {result_reg}, {result_reg}, #{operand2}")
            elif operand2.startswith('id'):
                assembly.append(f"LD{suffix} {result_reg}, {operand2}")
                assembly.append(f"DIV{suffix} {result_reg}, #{operand1}, {result_reg}")
            
            if not left.startswith('temp'):
                assembly.append(f"STR{suffix} {left}, {result_reg}")
        
        else:
            # Simple assignment: id1 = temp3 or id1 = id2 or id1 = 5
            if right.startswith('temp'):
                reg = f"R{temp_registers[right]}"
                assembly.append(f"STR{suffix} {left}, {reg}")
            elif right.startswith('id'):
                reg = "R1"
                assembly.append(f"LD{suffix} {reg}, {right}")
                assembly.append(f"STR{suffix} {left}, {reg}")
            else:
                # It's a constant number - direct store
                assembly.append(f"STR{suffix} {left}, #{right}")
    
    return assembly


# --- Pydantic Models ---
class TokenInfo(BaseModel):
    type: str
    value: str
    original: str


class LexicalResponse(BaseModel):
    tokens: List[TokenInfo]
    symbol_table: Dict[str, str]


class TreeNode(BaseModel):
    value: str
    node_type: str
    original_name: Optional[str] = None
    type_info: Optional[str] = None
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None


class CompileRequest(BaseModel):
    code: str = Field(..., description="Source code to compile", json_schema_extra={"example": "Z = 2 * y + 2.9 * X"})
    type_table: Dict[str, str] = Field(
        default_factory=dict,
        description="Variable type definitions",
        json_schema_extra={"example": {"y": "int", "X": "float"}}
    )


class CompileResponse(BaseModel):
    success: bool
    lexical: Dict
    syntax_tree: Dict
    semantic_tree: Dict
    intermediate_code: List[str]
    optimized_code: List[str]
    assembly_code: List[str]


class LexicalRequest(BaseModel):
    code: str = Field(..., description="Source code to analyze", json_schema_extra={"example": "Z = 2 * y + 2.9 * X"})


class LexicalOnlyResponse(BaseModel):
    success: bool
    tokens: List[TokenInfo]
    symbol_table: Dict[str, str]


class ErrorResponse(BaseModel):
    success: bool = False
    error: str


class HealthResponse(BaseModel):
    status: str
    service: str


# --- Hybrid Pydantic Models ---
class HybridRequest(BaseModel):
    code: str = Field(..., description="Source code to process", json_schema_extra={"example": "Z = 2 * y + 2.9 * X"})
    type_table: Dict[str, str] = Field(
        default_factory=dict,
        description="Variable type definitions",
        json_schema_extra={"example": {"y": "int", "X": "float"}}
    )
    value_table: Dict[str, float] = Field(
        default_factory=dict,
        description="Variable values for direct execution",
        json_schema_extra={"example": {"y": 5, "X": 3}}
    )


class HybridResponse(BaseModel):
    success: bool
    lexical: Dict
    syntax_tree: Dict
    semantic_tree: Dict
    execution_tree: Dict


# --- API Endpoints ---
@app.post('/api/compile', response_model=CompileResponse, responses={400: {"model": ErrorResponse}})
async def compile_code(request: CompileRequest):
    """
    Process code through all compilation phases (lexical, syntax, semantic, ICG, optimization, code generation).
    
    Returns all compilation steps for frontend visualization including:
    - Tokens and symbol table from lexical analysis
    - Syntax tree structure
    - Semantic tree with type coercion
    - Intermediate code generation steps
    - Optimized code
    - Assembly code
    """
    try:
        # Step 1: Lexical Analysis
        tokens, symbol_table = lexer(request.code)
        
        # Step 2: Syntax Analysis
        syntax_tree = Parser(tokens).parse()
        
        # Step 3: Semantic Analysis
        semantic_tree = Parser(tokens).parse()
        semantic_analysis(semantic_tree, request.type_table)
        
        # Step 4: Intermediate Code Generation
        icg_tree = Parser(tokens).parse()
        semantic_analysis(icg_tree, request.type_table)
        
        conversions = {}
        temp_counter = collect_conversions(icg_tree, request.type_table, conversions, 1, skip_left_assign=True)
        
        instructions = []
        for key, (temp_name, instruction) in sorted(conversions.items(), key=lambda x: x[1][0]):
            instructions.append(instruction)
        
        generate_icg(icg_tree, request.type_table, instructions, temp_counter, conversions)
        
        # Step 5: Code Optimization
        optimized_instructions = optimize_code(instructions, conversions)
        
        # Step 6: Code Generation (Assembly)
        assembly_code = generate_assembly(optimized_instructions, request.type_table)
        
        # Return all compilation phases
        return {
            'success': True,
            'lexical': {
                'tokens': [{'type': t[0], 'value': t[1], 'original': t[2] if len(t) > 2 else t[1]} for t in tokens],
                'symbol_table': symbol_table
            },
            'syntax_tree': syntax_tree.to_dict(),
            'semantic_tree': semantic_tree.to_dict(),
            'intermediate_code': instructions,
            'optimized_code': optimized_instructions,
            'assembly_code': assembly_code
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/api/lexical', response_model=LexicalOnlyResponse, responses={400: {"model": ErrorResponse}})
async def lexical_analysis(request: LexicalRequest):
    """
    Perform lexical analysis only.
    
    Returns tokens and symbol table without further compilation steps.
    """
    try:
        tokens, symbol_table = lexer(request.code)
        
        return {
            'success': True,
            'tokens': [{'type': t[0], 'value': t[1], 'original': t[2] if len(t) > 2 else t[1]} for t in tokens],
            'symbol_table': symbol_table
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get('/api/health', response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify API status"""
    return {
        'status': 'healthy',
        'service': 'compiler-api'
    }


# --- Hybrid API Endpoints ---
@app.post('/api/hybrid/lexical', response_model=LexicalOnlyResponse, responses={400: {"model": ErrorResponse}})
async def hybrid_lexical_analysis(request: LexicalRequest):
    """
    Perform hybrid lexical analysis (uses V1, V2 instead of id1, id2 and "is" instead of "=").
    """
    try:
        tokens, symbol_table = hybrid_lexer(request.code)
        
        return {
            'success': True,
            'tokens': [{'type': t[0], 'value': t[1], 'original': t[2] if len(t) > 2 else t[1]} for t in tokens],
            'symbol_table': symbol_table
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/api/hybrid/compile', response_model=HybridResponse, responses={400: {"model": ErrorResponse}})
async def hybrid_compile(request: HybridRequest):
    """
    Process code through hybrid approach (4 phases):
    1. Lexical Analysis (with V1, V2 naming and "is" operator)
    2. Syntax Analysis
    3. Semantic Analysis (with type coercion)
    4. Direct Execution (evaluate tree with user-provided values)
    """
    try:
        # Step 1: Hybrid Lexical Analysis
        tokens, symbol_table = hybrid_lexer(request.code)
        
        # Step 2: Syntax Analysis
        syntax_tree = Parser(tokens).parse()
        
        # Step 3: Semantic Analysis
        semantic_tree = Parser(tokens).parse()
        semantic_analysis(semantic_tree, request.type_table)
        
        # Step 4: Direct Execution
        execution_tree = Parser(tokens).parse()
        semantic_analysis(execution_tree, request.type_table)
        direct_execution(execution_tree, request.value_table, request.type_table)
        
        return {
            'success': True,
            'lexical': {
                'tokens': [{'type': t[0], 'value': t[1], 'original': t[2] if len(t) > 2 else t[1]} for t in tokens],
                'symbol_table': symbol_table
            },
            'syntax_tree': syntax_tree.to_dict(),
            'semantic_tree': semantic_tree.to_dict(),
            'execution_tree': NodeWithExecution.to_dict_with_execution(execution_tree)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == '__main__':
    uvicorn.run('compiler_api:app', host='0.0.0.0', port=5000, reload=True)
