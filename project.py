import re

tokens = {
    "NUMBER": r'\d+(\.\d+)?',
    "OP": r'[+\-*/<=(<>)]',
    "ID": r'[A-Za-z_]\w*',
    # "MISMATCH": r'.'
}

def main():
    code = input("Input: ")

    token_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in tokens.items())
    # print(token_regex)
    
    output_tokens = []
    symbols = {}
    counter = 1
    for match in re.finditer(token_regex, code):
        kind = match.lastgroup
        value = match.group()

        if kind == "NUMBER":
            value = float(value) if '.' in value else int(value)
            output_tokens.append(str(value))
        elif kind == "ID":
            if value not in symbols:
                symbols[value] = f"Id{counter}"
                counter += 1
            output_tokens.append(symbols[value])          
        elif kind == "OP":
            output_tokens.append(value)
        # elif kind == "MISMATCH":
        #     print(f"Unexpected character: {value}")
        #     exit(1)

    print("Output:", " ".join(output_tokens))
    print("Symbols:", symbols)

main()