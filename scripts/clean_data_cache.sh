#!/bin/bash

# use safe bash settings
set -euf -o pipefail

find ./data/hf -type f -name 'cache*.arrow' -delete
find ./data/processed -type f -name 'cache*.arrow' -delete
