#!/usr/bin/env python3
"""Mark all family subtasks with subtask=True in config.py."""

import re

# Read the generated groups to get all subtask names
with open('/tmp/family_groups_clean.txt', 'r') as f:
    content = f.read()

# Extract all benchmark names from the groups
subtask_names = set()
for line in content.split('\n'):
    match = re.search(r'"([^"]+)",\s*$', line.strip())
    if match:
        subtask_names.add(match.group(1))

print(f"Found {len(subtask_names)} subtasks to mark")

# Read config.py
with open('src/openbench/config.py', 'r') as f:
    lines = f.readlines()

# Process line by line, looking for benchmark entries
modified_count = 0
output_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    output_lines.append(line)

    # Check if this line starts a BenchmarkMetadata for a subtask
    for subtask in subtask_names:
        if f'"{subtask}": BenchmarkMetadata(' in line:
            # Found a subtask! Now find its closing paren
            # Look ahead to find the closing )
            j = i + 1
            while j < len(lines) and '),' not in lines[j]:
                output_lines.append(lines[j])
                j += 1

            if j < len(lines):
                # Found closing paren line
                closing_line = lines[j]
                # Check if subtask= is already present
                metadata_block = ''.join(lines[i:j+1])
                if 'subtask=' not in metadata_block:
                    # Add subtask=True before the closing paren
                    # Replace ),  with subtask=True,\n    ),
                    indented_line = closing_line.replace('),', '        subtask=True,\n    ),')
                    output_lines.append(indented_line)
                    modified_count += 1
                else:
                    output_lines.append(closing_line)

                i = j + 1
                break
    else:
        i += 1

print(f"Modified {modified_count} benchmarks")

# Write back
with open('src/openbench/config.py', 'w') as f:
    f.writelines(output_lines)

print("✅ Done marking subtasks")
