#!/usr/bin/env python3
"""
Script to add warmup property to all indicator classes.

Adds @property def warmup(self) -> int method to each indicator class
with explicit warmup calculation logic based on indicator type.
"""

import os
import re
from pathlib import Path


# Warmup calculation templates based on indicator type
WARMUP_TEMPLATES = {
    # Exact class name matches
    'RsiMom': 'return self.period * 10',
    'RsiDivergence': 'return self.rsi_period * 10 + self.pivot_window * 2 + self.lookback',
    'MacdDivergence': 'return self.slow * 5 + self.pivot_window * 2 + self.lookback',
    'MacdMom': 'return self.slow * 5',
    'PpoMom': 'return self.slow * 5',
    'TrixMom': 'return self.period * 12',
    'TsiMom': 'return self.period * 8',
    'T3Smooth': 'return self.period * 6',
    'StochRsiOsc': 'return self.rsi_period * 6 + self.stoch_period * 3',

    # Pattern-based matches (will use regex)
    'Stoch.*Osc': 'return (self.k_period + self.d_period + getattr(self, "smooth_k", 1)) * 3',
    'Adx.*': 'return self.period * 5',
    'Di.*': 'return self.period * 5',
    'Atr.*': 'return self.period * 5',
    'Natr.*': 'return self.period * 5',
    '.*Bollinger.*': 'return self.period * 2',
    'Bb.*': 'return self.period * 2',
    'Kama.*': 'return self.period * 5',
    'Jma.*': 'return self.period * 5',
    'Vidya.*': 'return self.period * 5',
    'Ema.*': 'return self.period * 5',
    'Tema.*': 'return self.period * 5',
    'Dema.*': 'return self.period * 5',
    'Zlma.*': 'return self.period * 5',
    'Hma.*': 'return self.period * 5',
    'Hull.*': 'return self.period * 5',
    'Rma.*': 'return self.period * 10',
    'Wma.*': 'return self.period',
    'Sma.*': 'return self.period',
    'Alma.*': 'return self.period * 5',
    'Frama.*': 'return self.period * 5',
    'McGinley.*': 'return self.period * 5',
    'Ssf.*': 'return self.period * 5',

    # Default for other indicators with period
    'default_period': 'return getattr(self, "period", getattr(self, "length", getattr(self, "window", 20))) * 5',
}


def get_warmup_logic(class_name: str) -> str:
    """Determine warmup logic for a given class name."""
    # Try exact match first
    if class_name in WARMUP_TEMPLATES:
        return WARMUP_TEMPLATES[class_name]

    # Try pattern matches
    for pattern, logic in WARMUP_TEMPLATES.items():
        if pattern.startswith('.*') or '.*' in pattern or pattern.endswith('.*'):
            if re.match(pattern, class_name):
                return logic

    # Default: try to use period attribute
    return WARMUP_TEMPLATES['default_period']


def add_warmup_to_file(file_path: Path) -> bool:
    """Add warmup property to all classes in a file."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Find all class definitions
    # Match: class ClassName(BaseClass):
    # Regardless of decorators above it
    class_pattern = r'^class\s+(\w+)\([^)]+\):$'

    modified = False
    new_content = content

    for match in re.finditer(class_pattern, content, re.MULTILINE):
        class_name = match.group(1)

        # Check if warmup property already exists
        class_start = match.end()
        # Find the next class or end of file
        next_class = re.search(r'\n@[\w_]+|^class ', content[class_start:], re.MULTILINE)
        if next_class:
            class_body = content[class_start:class_start + next_class.start()]
        else:
            class_body = content[class_start:]

        # Skip if warmup property already exists
        if '@property' in class_body and 'def warmup' in class_body:
            print(f"  {class_name}: already has warmup property, skipping")
            continue

        # Find where to insert (after test_params or after compute methods)
        # Look for the last method definition in the class
        insert_pos = None

        # Find all method definitions
        method_matches = list(re.finditer(r'\n    def \w+\([^)]*\).*?(?=\n    def |\n\n[^ ]|\Z)',
                                          class_body, re.DOTALL))

        if method_matches:
            last_method = method_matches[-1]
            insert_pos = class_start + last_method.end()
        else:
            # No methods found, insert before class ends
            # Find the first line that's not indented (next class or module level)
            next_unindented = re.search(r'\n[^\s#]', class_body)
            if next_unindented:
                insert_pos = class_start + next_unindented.start()
            else:
                insert_pos = len(content)

        # Generate warmup property
        warmup_logic = get_warmup_logic(class_name)
        warmup_property = f'''

    @property
    def warmup(self) -> int:
        """Minimum bars needed for stable, reproducible output."""
        {warmup_logic}
'''

        # Insert the property
        new_content = new_content[:insert_pos] + warmup_property + new_content[insert_pos:]
        modified = True
        print(f"  {class_name}: added warmup property")

        # Update content for next iteration
        content = new_content

    if modified:
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True

    return False


def main():
    # Find all Python files in ta module
    ta_dir = Path('/home/alastor/signalflow-ta/src/signalflow/ta')

    python_files = []
    for root, dirs, files in os.walk(ta_dir):
        # Skip test directories
        if 'test' in root or '__pycache__' in root:
            continue

        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                python_files.append(Path(root) / file)

    print(f"Found {len(python_files)} Python files")
    print()

    modified_count = 0
    for file_path in sorted(python_files):
        rel_path = file_path.relative_to(ta_dir)
        print(f"Processing {rel_path}...")

        try:
            if add_warmup_to_file(file_path):
                modified_count += 1
        except Exception as e:
            print(f"  ERROR: {e}")

    print()
    print(f"Modified {modified_count} files")


if __name__ == '__main__':
    main()
