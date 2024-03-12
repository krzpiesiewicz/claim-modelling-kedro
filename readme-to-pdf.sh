#!/bin/bash
echo -e "> source venv/bin/activate"
source venv/bin/activate

echo -e "> md2pdf README.md --theme=github"
md2pdf README.md --theme=github
