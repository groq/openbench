#!/bin/bash
# Find all directories and set permissions to 755
find gist.github.com -type d -exec chmod 755 {} +

# Find all regular files and set permissions to 644
find gist.github.com -type f -exec chmod 644 {} +
