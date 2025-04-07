#!/bin/bash

MESSAGE=$1

if [ -z "$MESSAGE" ]; then
  echo "Please provide a commit message."
  exit 1
fi

git add .
git commit -m "$MESSAGE"
git push origin main
