#!/usr/bin/env python3

target_file='lib/custom.js'

with open(target_file, 'rt') as f:
    content = f.read()

content = content.replace(
    '\tbottommost = Math.max(1, levels.length - 1)',
    '\t//bottommost = Math.max(1, levels.length - 1)\n\tbottommost = 1')

with open(target_file, 'wt') as f:
    f.write(content)
    print(f'{target_file} saved.')
