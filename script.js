const API_URL = 'http://localhost:5000/api';

// ============== COMPILER TAB DOM Elements ==============
const codeInput = document.getElementById('code-input');
const analyzeBtn = document.getElementById('analyze-btn');
const compileBtn = document.getElementById('compile-btn');
const resetBtn = document.getElementById('reset-btn');
const typeTableSection = document.getElementById('type-table-section');
const typeInputs = document.getElementById('type-inputs');
const tokensOutput = document.getElementById('tokens-output');
const symbolTableBody = document.querySelector('#symbol-table tbody');
const syntaxTree = document.getElementById('syntax-tree');
const semanticTree = document.getElementById('semantic-tree');
const icgOutput = document.getElementById('icg-output');
const optimizedOutput = document.getElementById('optimized-output');
const assemblyOutput = document.getElementById('assembly-output');
const errorDisplay = document.getElementById('error-display');

// ============== HYBRID TAB DOM Elements ==============
const hybridCodeInput = document.getElementById('hybrid-code-input');
const hybridAnalyzeBtn = document.getElementById('hybrid-analyze-btn');
const hybridCompileBtn = document.getElementById('hybrid-compile-btn');
const hybridResetBtn = document.getElementById('hybrid-reset-btn');
const hybridTypeTableSection = document.getElementById('hybrid-type-table-section');
const hybridTypeInputs = document.getElementById('hybrid-type-inputs');
const hybridValueTableSection = document.getElementById('hybrid-value-table-section');
const hybridValueInputs = document.getElementById('hybrid-value-inputs');
const hybridTokensOutput = document.getElementById('hybrid-tokens-output');
const hybridSymbolTableBody = document.querySelector('#hybrid-symbol-table tbody');
const hybridSyntaxTree = document.getElementById('hybrid-syntax-tree');
const hybridSemanticTree = document.getElementById('hybrid-semantic-tree');
const hybridExecutionTree = document.getElementById('hybrid-execution-tree');

// ============== TAB Navigation ==============
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Remove active from all
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        
        // Add active to clicked
        btn.classList.add('active');
        const tabId = btn.dataset.tab + '-tab';
        document.getElementById(tabId).classList.add('active');
    });
});

let currentSymbolTable = {};
let hybridSymbolTable = {};

// ============== COMPILER TAB Event Listeners ==============
analyzeBtn.addEventListener('click', analyzeLexical);
compileBtn.addEventListener('click', compile);
resetBtn.addEventListener('click', reset);
codeInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') analyzeLexical();
});

// ============== HYBRID TAB Event Listeners ==============
hybridAnalyzeBtn.addEventListener('click', hybridAnalyzeLexical);
hybridCompileBtn.addEventListener('click', hybridCompile);
hybridResetBtn.addEventListener('click', hybridReset);
hybridCodeInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') hybridAnalyzeLexical();
});

// Analyze lexical (first step)
async function analyzeLexical() {
    const code = codeInput.value.trim();
    if (!code) {
        showError('Please enter some code to analyze.');
        return;
    }

    hideError();
    clearOutputs();

    try {
        const response = await fetch(`${API_URL}/lexical`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code })
        });

        const data = await response.json();

        if (!data.success) {
            showError(data.detail || data.error || 'Analysis failed');
            return;
        }

        // Display tokens
        displayTokens(data.tokens);

        // Display symbol table
        displaySymbolTable(data.symbol_table);

        // Store symbol table and show type inputs
        currentSymbolTable = data.symbol_table;
        showTypeInputs(data.symbol_table);

        // Enable compile button
        compileBtn.disabled = false;

    } catch (error) {
        showError(`Error connecting to API: ${error.message}`);
    }
}

// Full compilation
async function compile() {
    const code = codeInput.value.trim();
    const typeTable = getTypeTable();

    hideError();

    try {
        const response = await fetch(`${API_URL}/compile`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code, type_table: typeTable })
        });

        const data = await response.json();

        if (!data.success) {
            showError(data.detail || data.error || 'Compilation failed');
            return;
        }

        // Display all phases
        displayTokens(data.lexical.tokens);
        displaySymbolTable(data.lexical.symbol_table);
        displaySyntaxTree(data.syntax_tree);
        displaySemanticTree(data.semantic_tree);
        displayICG(data.intermediate_code);
        displayOptimized(data.optimized_code);
        displayAssembly(data.assembly_code);

    } catch (error) {
        showError(`Error connecting to API: ${error.message}`);
    }
}

// Display functions
function displayTokens(tokens) {
    tokensOutput.innerHTML = tokens.map(t => 
        `<span class="token ${t.type}">${t.type === 'ID' ? t.value : t.original}</span>`
    ).join('');
}

function displaySymbolTable(symbolTable) {
    symbolTableBody.innerHTML = Object.entries(symbolTable).map(([name, id]) =>
        `<tr><td>${name}</td><td>${id}</td></tr>`
    ).join('');
}

function showTypeInputs(symbolTable) {
    const variables = Object.keys(symbolTable);
    
    // Get the first variable (the one being assigned to) - it shouldn't have a type selector
    const code = codeInput.value.trim();
    const assignedVar = code.split('=')[0].trim();
    
    // Filter out the assigned variable
    const inputVariables = variables.filter(v => v !== assignedVar);
    
    if (inputVariables.length === 0) {
        typeTableSection.style.display = 'none';
        return;
    }

    typeInputs.innerHTML = inputVariables.map(varName => `
        <div class="type-input-group">
            <label>${varName}:</label>
            <select id="type-${varName}">
                <option value="int">int</option>
                <option value="float">float</option>
            </select>
        </div>
    `).join('');

    typeTableSection.style.display = 'block';
}

function getTypeTable() {
    const typeTable = {};
    for (const varName of Object.keys(currentSymbolTable)) {
        const select = document.getElementById(`type-${varName}`);
        if (select) {
            typeTable[varName] = select.value;
        }
    }
    return typeTable;
}

function displaySyntaxTree(tree) {
    syntaxTree.innerHTML = buildTreeHTML(tree, false);
    // Draw lines after a small delay to ensure DOM is rendered
    setTimeout(() => drawTreeLines(syntaxTree), 50);
}

function displaySemanticTree(tree) {
    semanticTree.innerHTML = buildTreeHTML(tree, true);
    setTimeout(() => drawTreeLines(semanticTree), 50);
}

function buildTreeHTML(node, showCoercion = false) {
    if (!node) return '';

    let nodeClass = 'tree-node-value';
    let displayValue = node.value;

    // Determine node styling
    if (node.node_type === 'OP' || node.node_type === 'ASSIGN') {
        nodeClass += ' operator';
    } else if (node.node_type === 'ID') {
        nodeClass += ' id';
        if (node.original_name) {
            displayValue = node.original_name;
        }
    } else if (node.node_type === 'NUMBER') {
        nodeClass += ' number';
    }

    // Add coercion indicator for semantic tree
    if (showCoercion && node.type_info === 'int to float') {
        nodeClass += ' coerced';
        displayValue += ' →float';
    }

    let html = `<div class="tree-node">`;
    html += `<div class="${nodeClass}">${displayValue}</div>`;

    if (node.left || node.right) {
        html += `<div class="tree-children">`;
        if (node.left) {
            html += `<div class="tree-child">${buildTreeHTML(node.left, showCoercion)}</div>`;
        }
        if (node.right) {
            html += `<div class="tree-child">${buildTreeHTML(node.right, showCoercion)}</div>`;
        }
        html += `</div>`;
    }

    html += `</div>`;
    return html;
}

function drawTreeLines(container) {
    // Remove existing SVG if any
    const existingSvg = container.querySelector('.tree-svg');
    if (existingSvg) existingSvg.remove();

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.classList.add('tree-svg');
    
    const containerRect = container.getBoundingClientRect();
    svg.setAttribute('width', container.scrollWidth);
    svg.setAttribute('height', container.scrollHeight);
    
    // Find all parent nodes with children
    const nodes = container.querySelectorAll('.tree-node');
    
    nodes.forEach(node => {
        const parentValue = node.querySelector(':scope > .tree-node-value');
        const childrenContainer = node.querySelector(':scope > .tree-children');
        
        if (parentValue && childrenContainer) {
            const children = childrenContainer.querySelectorAll(':scope > .tree-child > .tree-node > .tree-node-value');
            
            const parentRect = parentValue.getBoundingClientRect();
            const parentX = parentRect.left + parentRect.width / 2 - containerRect.left;
            const parentY = parentRect.bottom - containerRect.top;
            
            children.forEach(child => {
                const childRect = child.getBoundingClientRect();
                const childX = childRect.left + childRect.width / 2 - containerRect.left;
                const childY = childRect.top - containerRect.top;
                
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', parentX);
                line.setAttribute('y1', parentY);
                line.setAttribute('x2', childX);
                line.setAttribute('y2', childY);
                svg.appendChild(line);
            });
        }
    });
    
    container.insertBefore(svg, container.firstChild);
}

function displayICG(instructions) {
    icgOutput.textContent = instructions.join('\n');
}

function displayOptimized(instructions) {
    optimizedOutput.textContent = instructions.join('\n');
}

function displayAssembly(instructions) {
    assemblyOutput.textContent = instructions.join('\n');
}

// Utility functions
function showError(message) {
    errorDisplay.textContent = `❌ Error: ${message}`;
    errorDisplay.style.display = 'block';
}

function hideError() {
    errorDisplay.style.display = 'none';
}

function clearOutputs() {
    tokensOutput.innerHTML = '<span style="color:#666;">Click "Analyze Code" to see tokens...</span>';
    symbolTableBody.innerHTML = '';
    syntaxTree.innerHTML = '<span style="color:#666;">Compile to see parse tree...</span>';
    semanticTree.innerHTML = '<span style="color:#666;">Compile to see semantic tree...</span>';
    icgOutput.textContent = 'Compile to see intermediate code...';
    optimizedOutput.textContent = 'Compile to see optimized code...';
    assemblyOutput.textContent = 'Compile to see assembly code...';
}

function reset() {
    codeInput.value = '';
    currentSymbolTable = {};
    typeTableSection.style.display = 'none';
    typeInputs.innerHTML = '';
    compileBtn.disabled = true;
    clearOutputs();
    hideError();
}

// ============== HYBRID TAB Functions ==============

// Hybrid lexical analysis
async function hybridAnalyzeLexical() {
    const code = hybridCodeInput.value.trim();
    if (!code) {
        showError('Please enter some code to analyze.');
        return;
    }

    hideError();
    clearHybridOutputs();

    try {
        const response = await fetch(`${API_URL}/hybrid/lexical`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code })
        });

        const data = await response.json();

        if (!data.success) {
            showError(data.detail || data.error || 'Analysis failed');
            return;
        }

        // Display tokens
        displayHybridTokens(data.tokens);

        // Display symbol table
        displayHybridSymbolTable(data.symbol_table);

        // Store symbol table and show type/value inputs
        hybridSymbolTable = data.symbol_table;
        showHybridTypeInputs(data.symbol_table);
        showHybridValueInputs(data.symbol_table);

        // Enable execute button
        hybridCompileBtn.disabled = false;

    } catch (error) {
        showError(`Error connecting to API: ${error.message}`);
    }
}

// Hybrid compile (direct execution)
async function hybridCompile() {
    const code = hybridCodeInput.value.trim();
    const typeTable = getHybridTypeTable();
    const valueTable = getHybridValueTable();

    hideError();

    try {
        const response = await fetch(`${API_URL}/hybrid/compile`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code, type_table: typeTable, value_table: valueTable })
        });

        const data = await response.json();

        if (!data.success) {
            showError(data.detail || data.error || 'Execution failed');
            return;
        }

        // Display all phases
        displayHybridTokens(data.lexical.tokens);
        displayHybridSymbolTable(data.lexical.symbol_table);
        displayHybridSyntaxTree(data.syntax_tree);
        displayHybridSemanticTree(data.semantic_tree);
        displayHybridExecutionTree(data.execution_tree);

    } catch (error) {
        showError(`Error connecting to API: ${error.message}`);
    }
}

function displayHybridTokens(tokens) {
    hybridTokensOutput.innerHTML = tokens.map(t => 
        `<span class="token ${t.type}">${t.type === 'ID' ? t.value : t.original}</span>`
    ).join('');
}

function displayHybridSymbolTable(symbolTable) {
    hybridSymbolTableBody.innerHTML = Object.entries(symbolTable).map(([name, id]) =>
        `<tr><td>${name}</td><td>${id}</td></tr>`
    ).join('');
}

function showHybridTypeInputs(symbolTable) {
    const variables = Object.keys(symbolTable);
    
    // Get the assigned variable (left side of =)
    const code = hybridCodeInput.value.trim();
    const assignedVar = code.split('=')[0].trim();
    
    // Filter out the assigned variable
    const inputVariables = variables.filter(v => v !== assignedVar);
    
    if (inputVariables.length === 0) {
        hybridTypeTableSection.style.display = 'none';
        return;
    }

    hybridTypeInputs.innerHTML = inputVariables.map(varName => `
        <div class="type-input-group">
            <label>${varName}:</label>
            <select id="hybrid-type-${varName}">
                <option value="int">int</option>
                <option value="float">float</option>
            </select>
        </div>
    `).join('');

    hybridTypeTableSection.style.display = 'block';
}

function showHybridValueInputs(symbolTable) {
    const variables = Object.keys(symbolTable);
    
    // Get the assigned variable
    const code = hybridCodeInput.value.trim();
    const assignedVar = code.split('=')[0].trim();
    
    // Filter out the assigned variable
    const inputVariables = variables.filter(v => v !== assignedVar);
    
    if (inputVariables.length === 0) {
        hybridValueTableSection.style.display = 'none';
        return;
    }

    // Placeholder values
    const placeholders = [5, 3, 7, 2, 10];

    hybridValueInputs.innerHTML = inputVariables.map((varName, idx) => `
        <div class="value-input-group">
            <label>${varName}:</label>
            <input type="number" id="hybrid-value-${varName}" value="${placeholders[idx % placeholders.length]}" step="any">
        </div>
    `).join('');

    hybridValueTableSection.style.display = 'block';
}

function getHybridTypeTable() {
    const typeTable = {};
    for (const varName of Object.keys(hybridSymbolTable)) {
        const select = document.getElementById(`hybrid-type-${varName}`);
        if (select) {
            typeTable[varName] = select.value;
        }
    }
    return typeTable;
}

function getHybridValueTable() {
    const valueTable = {};
    for (const varName of Object.keys(hybridSymbolTable)) {
        const input = document.getElementById(`hybrid-value-${varName}`);
        if (input) {
            valueTable[varName] = parseFloat(input.value) || 0;
        }
    }
    return valueTable;
}

function displayHybridSyntaxTree(tree) {
    hybridSyntaxTree.innerHTML = buildTreeHTML(tree, false);
    setTimeout(() => drawTreeLines(hybridSyntaxTree), 50);
}

function displayHybridSemanticTree(tree) {
    hybridSemanticTree.innerHTML = buildTreeHTML(tree, true);
    setTimeout(() => drawTreeLines(hybridSemanticTree), 50);
}

function displayHybridExecutionTree(tree) {
    hybridExecutionTree.innerHTML = buildExecutionTreeHTML(tree);
    setTimeout(() => drawTreeLines(hybridExecutionTree), 50);
    
    // Display final answer
    displayFinalAnswer(tree);
}

function displayFinalAnswer(tree) {
    const finalAnswerDiv = document.getElementById('hybrid-final-answer');
    if (!tree || tree.computed_value === null || tree.computed_value === undefined) {
        finalAnswerDiv.innerHTML = '';
        return;
    }
    
    const val = tree.computed_value;
    const formatted = Number.isInteger(val) ? val : val.toFixed(2);
    
    // Get the V name (V1) and original variable name
    const vName = tree.left ? tree.left.value : 'V1';
    const code = hybridCodeInput.value.trim();
    const originalVar = code.split('=')[0].trim();
    
    finalAnswerDiv.innerHTML = `
        <div class="final-answer-line"><span class="var-name">${vName}</span> is <span class="result-value">${formatted}</span></div>
        <div class="final-answer-line"><span class="var-name">${originalVar}</span> = <span class="result-value">${formatted}</span></div>
    `;
}

function buildExecutionTreeHTML(node, isAssignedVar = false) {
    if (!node) return '';

    let nodeClass = 'tree-node-value';
    let displayValue = node.value;
    let computedDisplay = '';

    // Determine node styling
    if (node.node_type === 'OP' || node.node_type === 'ASSIGN') {
        nodeClass += ' operator';
    } else if (node.node_type === 'ID') {
        nodeClass += ' id';
    } else if (node.node_type === 'NUMBER') {
        nodeClass += ' number';
    }

    // Add coercion indicator
    if (node.type_info === 'int to float') {
        nodeClass += ' coerced';
    }

    // Show computed value (but not for the assigned variable V1)
    if (node.computed_value !== null && node.computed_value !== undefined && !isAssignedVar) {
        // Format number nicely
        const val = node.computed_value;
        const formatted = Number.isInteger(val) ? val : val.toFixed(2);
        computedDisplay = `<span class="computed-value">= ${formatted}</span>`;
    }

    let html = `<div class="tree-node">`;
    html += `<div class="${nodeClass}">${displayValue}${computedDisplay}</div>`;

    if (node.left || node.right) {
        html += `<div class="tree-children">`;
        if (node.left) {
            // If this is ASSIGN node, left child is V1 (assigned var)
            const leftIsAssigned = node.node_type === 'ASSIGN';
            html += `<div class="tree-child">${buildExecutionTreeHTML(node.left, leftIsAssigned)}</div>`;
        }
        if (node.right) {
            html += `<div class="tree-child">${buildExecutionTreeHTML(node.right, false)}</div>`;
        }
        html += `</div>`;
    }

    html += `</div>`;
    return html;
}

function clearHybridOutputs() {
    hybridTokensOutput.innerHTML = '<span style="color:#666;">Click "Analyze Code" to see tokens...</span>';
    hybridSymbolTableBody.innerHTML = '';
    hybridSyntaxTree.innerHTML = '<span style="color:#666;">Execute to see parse tree...</span>';
    hybridSemanticTree.innerHTML = '<span style="color:#666;">Execute to see semantic tree...</span>';
    hybridExecutionTree.innerHTML = '<span style="color:#666;">Execute to see evaluated tree...</span>';
    document.getElementById('hybrid-final-answer').innerHTML = '';
}

function hybridReset() {
    hybridCodeInput.value = '';
    hybridSymbolTable = {};
    hybridTypeTableSection.style.display = 'none';
    hybridValueTableSection.style.display = 'none';
    hybridTypeInputs.innerHTML = '';
    hybridValueInputs.innerHTML = '';
    hybridCompileBtn.disabled = true;
    clearHybridOutputs();
    hideError();
}

// Initial state
clearOutputs();
clearHybridOutputs();
